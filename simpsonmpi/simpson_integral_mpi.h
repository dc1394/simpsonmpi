/*! \file simpson_integral.h
    \brief MPIで並列化してsimpsonの公式で数値積分を行うクラスの宣言と実装

    Copyright ©  2016 @dc1394 All Rights Reserved.
*/
#ifndef _SIMPSON_INTEGRAL_MPI_H_
#define _SIMPSON_INTEGRAL_MPI_H_

#pragma once

#include "functional.h"
#include <cstdint>                      // for std::int32_t
#include <mpi.h>
#include <boost/numeric/interval.hpp>   // boost::numeric::interval

namespace simpsonmpi {
    template <typename FUNCTYPE>
    class SimpsonMpi final {
        // #region コンストラクタ・デストラクタ

    public:
        //! A constructor.
        /*!
            唯一のコンストラクタ
            \param n simpsonの公式の分点
        */
        SimpsonMpi(myfunctional::Functional<FUNCTYPE> const & func, std::int32_t n, double x1, double x2)
            : func_(func), n_(n), x1_(x1), x2_(x2), dh_((x2_ - x1_) / static_cast<double>(n_)) {}

        //! A destructor.
        /*!
            デフォルトデストラクタ
        */
        ~SimpsonMpi() = default;
        
        // #endregion コンストラクタ・デストラクタ
        
        // #region publicメンバ関数
                
        void operator()() const;

        // #endregion publicメンバ関数

        // #region privateメンバ関数

    private:
        //! A private member function (const).
        /*!
            プロセスのランクと全体の閾値を指定して、プロセスの担当領域を決定する
            \param pnum 全プロセス数
            \param rank プロセスのランク
            \return 該当プロセスが積分の和を求める区間
        */
        boost::numeric::interval<std::int32_t> assign(std::int32_t pnum, std::int32_t rank) const;
        
        //! A private member function (const).
        /*!
            指定した区間について、Simpsonの公式で総和を求める
            \param interval 該当プロセスが和を求める区間
            \return 積分値
        */
        double simpson(boost::numeric::interval<std::int32_t> const & interval) const;
        
        // #endregion privateメンバ関数

        // #region メンバ変数
        
        //! A private member variable (constant).
        /*!
            被積分関数
        */
        myfunctional::Functional<FUNCTYPE> const func_;

        //! A private member variable (constant).
        /*!
            simpsonの公式の積分点
        */
        std::int32_t const n_;

        //! A private member variable (constant).
        /*!
            積分の下端
        */
        double const x1_;

        //! A private member variable (constant).
        /*!
            積分の上端
        */
        double const x2_;
        
        //! A private member variable (constant).
        /*!
            積分の微小区間
        */
        double const dh_;
        
        // #endregion メンバ変数

        // #region 禁止されたコンストラクタ・メンバ関数

        //! A private constructor (deleted).
        /*!
            デフォルトコンストラクタ（禁止）
        */
        SimpsonMpi() = delete;

        //! A private copy constructor (deleted).
        /*!
            コピーコンストラクタ（禁止）
        */
        SimpsonMpi(SimpsonMpi const &) = delete;

        //! A private member function (deleted).
        /*!
            operator=()の宣言（禁止）
            \param コピー元のオブジェクト
            \return コピー元のオブジェクト
        */
        SimpsonMpi & operator=(SimpsonMpi const &) = delete;

        // #endregion 禁止されたコンストラクタ・メンバ関数
    };

    // #region template publicメンバ関数の実装

    template <typename FUNCTYPE>
    void SimpsonMpi<FUNCTYPE>::operator()() const
    {
        MPI_Init(nullptr, nullptr);

        std::int32_t my_rank;                       // 自分のプロセスランク
                                                    // rank取得
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        std::int32_t p_num;                         // プロセス数
                                                    // プロセス数取得
        MPI_Comm_size(MPI_COMM_WORLD, &p_num);

        std::int32_t localnmin;                     // プロセッサ毎の閾値の分割数
        std::int32_t localnmax;                     // プロセッサ毎の閾値の分割数

        auto const start = MPI_Wtime();             // 時間測定開始

                                                    // プロセス毎の閾値の算出及び積分
        auto local_result = simpson(assign(p_num, my_rank));

        double result;                              // 全体の積分値
        std::int32_t dest = 0;
        std::int32_t tag = 0;

        // 各プロセッサの積分結果を収集
        if (!my_rank) {
            result = local_result;
            for (auto source = 1; source < p_num; source++) {
                MPI_Status status;
                // 積分結果を収集
                MPI_Recv(&local_result, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);

                result += local_result;
            }
        }
        else {
            // プロセス0以外はプロセス0へ結果を送信
            MPI_Send(&local_result, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }

        if (!my_rank) {
            result *= (dh_ / 3.0);
            fprintf(stdout, "TOTAL :: x = [%f,%f] : n = %d : S = %.15f\n", 1.0, 4.0, N, result);
            auto const finish = MPI_Wtime();        // 時間測定終了

            printf("Elapsed time = %f[sec]\n", finish - start);
        }

        MPI_Finalize();
    }

    // #endregion template publicメンバ関数の実装
    
    // #region template privateメンバ関数の実装

    template <typename FUNCTYPE>
    boost::numeric::interval<std::int32_t> SimpsonMpi<FUNCTYPE>::assign(std::int32_t pnum, std::int32_t rank) const
    {
        auto const nmax = n_ / 2;
        std::int32_t localnmax;
        if (rank == pnum - 1) {
            localnmax = nmax;
        }
        else {
            localnmax = nmax / pnum * (rank + 1);
        }

        return boost::numeric::interval<std::int32_t>(nmax / pnum * rank, localnmax);
    }

    template <typename FUNCTYPE>
    double SimpsonMpi<FUNCTYPE>::simpson(boost::numeric::interval<std::int32_t> const & interval) const
    {
        auto sum = 0.0;

        for (auto i = interval.lower(); i < interval.upper(); i++) {
            auto const f0 = func_(x1_ + static_cast<double>(2 * i) * dh_);
            auto const f1 = func_(x1_ + static_cast<double>(2 * i + 1) * dh_);
            auto const f2 = func_(x1_ + static_cast<double>(2 * i + 2) * dh_);
            sum += (f0 + 4.0 * f1 + f2);
        }

        return sum;
    }

    // #endregion template privateメンバ関数の実装
}

#endif  // _SIMPSON_INTEGRAL_MPI_H_
