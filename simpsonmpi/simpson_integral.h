#include "functional.h"
#include <cstdint>
#include <tuple>
#include <mpi.h>

namespace simpsonmpi {
    template <typename FUNCTYPE>
    class SimpsonMpi final {
    public:
        // #region コンストラクタ

        //! A constructor.
        /*!
            唯一のコンストラクタ
            \param n simpsonの公式の分点
        */
        SimpsonMpi::SimpsonMpi(myfunctional::Functional<FUNCTYPE> const & func, std::int32_t n, double x1, double x2);

        // #endregion コンストラクタ

        // #region publicメンバ関数
                
        void operator()() const;

        // #endregion publicメンバ関数

    private:

        // #region privateメンバ関数

        //! A private member function.
        /*!
            プロセスのランクと全体の閾値を指定して、プロセスの担当領域を決定する
            \param pnum 全プロセス数
            \param rank プロセスのランク
            \param xmin 閾値全体の最小値
            \param xmax 閾値全体の最大値
            \return プロセスの担当分割数、プロセスの担当閾値の最小値、プロセスの担当閾値の最大値の組のstd::tuple
        */
        std::tuple<std::int32_t, double, double> assign(std::int32_t pnum, std::int32_t rank, double xmin, double xmax) const;
        
        //! A public member function (template function).
        /*!
            積分対象関数の積分値を閾値[x1, x2]をn個の領域に分割しSimpsonの公式によって数値積分を実行する
            \param n 分割数
            \param x1 積分の下端
            \param x2 積分の上端
            \return 積分値
        */
        double simpson(std::int32_t n, double x1, double x2) const;
        
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

    template <typename FUNCTYPE>
    SimpsonMpi<FUNCTYPE>::SimpsonMpi(myfunctional::Functional<FUNCTYPE> const & func, std::int32_t n, double x1, double x2)
        : func_(func), n_(n), x1_(x1), x2_(x2)
    {
    }

    template <typename FUNCTYPE>
    std::tuple<std::int32_t, double, double> SimpsonMpi<FUNCTYPE>::assign(std::int32_t pnum, std::int32_t rank, double xmin, double xmax) const
    {
        auto const localn = n_ / pnum;
        auto const h = (xmax - xmin) / static_cast<double>(n_);
        auto const localxmin = xmin + rank * localn * h;
        auto const localxmax = localxmin + localn * h;

        return std::make_tuple(localn, localxmin, localxmax);
    }

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
                
        std::int32_t localn;                        // プロセッサ毎の閾値の分割数
        double localxmax;                           // プロセッサ毎の閾値の最大値	
        double localxmin;                           // プロセッサ毎の閾値の最小値

        // プロセス毎の閾値の算出
        std::tie(localn, localxmin, localxmax) = assign(p_num, my_rank, x1_, x2_);

        // プロセス毎に積分
        auto local_result = simpson(localn, localxmin, localxmax);
        
        double result;                              // 全体の積分値
        std::int32_t dest = 0;
        std::int32_t tag = 0;

        // 各プロセッサの積分結果を収集
        if (!my_rank) {
            // プロセス0は全ての他プロセスから結果を受信
            fprintf(stdout, "P%d :: x = [%f,%f] : n = %d : s = %f\n", 0,
                localxmin, localxmax, localn, local_result);
            result = local_result;
            for (auto source = 1; source < p_num; source++)
            {
                MPI_Status status;
                // 積分結果を収集
                MPI_Recv(&local_result, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                
                // プロセス毎の結果出力
                std::tie(localn, localxmin, localxmax) = assign(p_num, source, x1_, x2_);
                
                fprintf(stdout, "P%d :: x = [%f,%f] : n = %d : s = %f\n",
                    source, localxmin, localxmax, localn,
                    local_result);

                result += local_result;
            }
        }
        else {
            // プロセス0以外はプロセス0へ結果を送信
            MPI_Send(&local_result, 1, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }

        if (!my_rank) {
            fprintf(stdout, "TOTAL :: x = [%f,%f] : n = %d : S = %f\n", 1.0, 4.0, N, result);
        }

        MPI_Finalize();
    }

    template <typename FUNCTYPE>
    double SimpsonMpi<FUNCTYPE>::simpson(std::int32_t n, double x1, double x2) const
    {
        auto sum = 0.0;
        auto const dh = (x2 - x1) / static_cast<double>(n);

        for (auto i = 0; i < n; i += 2) {
            auto const f0 = func_(x1 + static_cast<double>(i) * dh);
            auto const f1 = func_(x1 + static_cast<double>(i + 1) * dh);
            auto const f2 = func_(x1 + static_cast<double>(i + 2) * dh);
            sum += (f0 + 4.0 * f1 + f2);
        }

        return sum * dh / 3.0;
    }
}