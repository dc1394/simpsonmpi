#include "functional.h"
#include <cstdint>
#include <tuple>
#include <mpi.h>

namespace simpsonmpi {
    template <typename FUNCTYPE>
    class SimpsonMpi final {
    public:
        // #region �R���X�g���N�^

        //! A constructor.
        /*!
            �B��̃R���X�g���N�^
            \param n simpson�̌����̕��_
        */
        SimpsonMpi::SimpsonMpi(myfunctional::Functional<FUNCTYPE> const & func, std::int32_t n, double x1, double x2);

        // #endregion �R���X�g���N�^

        // #region public�����o�֐�
                
        void operator()() const;

        // #endregion public�����o�֐�

    private:

        // #region private�����o�֐�

        //! A private member function.
        /*!
            �v���Z�X�̃����N�ƑS�̂�臒l���w�肵�āA�v���Z�X�̒S���̈�����肷��
            \param pnum �S�v���Z�X��
            \param rank �v���Z�X�̃����N
            \param xmin 臒l�S�̂̍ŏ��l
            \param xmax 臒l�S�̂̍ő�l
            \return �v���Z�X�̒S���������A�v���Z�X�̒S��臒l�̍ŏ��l�A�v���Z�X�̒S��臒l�̍ő�l�̑g��std::tuple
        */
        std::tuple<std::int32_t, double, double> assign(std::int32_t pnum, std::int32_t rank, double xmin, double xmax) const;
        
        //! A public member function (template function).
        /*!
            �ϕ��Ώۊ֐��̐ϕ��l��臒l[x1, x2]��n�̗̈�ɕ�����Simpson�̌����ɂ���Đ��l�ϕ������s����
            \param n ������
            \param x1 �ϕ��̉��[
            \param x2 �ϕ��̏�[
            \return �ϕ��l
        */
        double simpson(std::int32_t n, double x1, double x2) const;
        
        // #endregion private�����o�֐�

        // #region �����o�ϐ�
        
        //! A private member variable (constant).
        /*!
            ��ϕ��֐�
        */
        myfunctional::Functional<FUNCTYPE> const func_;

        //! A private member variable (constant).
        /*!
            simpson�̌����̐ϕ��_
        */
        std::int32_t const n_;

        //! A private member variable (constant).
        /*!
            �ϕ��̉��[
        */
        double const x1_;

        //! A private member variable (constant).
        /*!
            �ϕ��̏�[
        */
        double const x2_;
        
        // #endregion �����o�ϐ�

        // #region �֎~���ꂽ�R���X�g���N�^�E�����o�֐�

        //! A private constructor (deleted).
        /*!
            �f�t�H���g�R���X�g���N�^�i�֎~�j
        */
        SimpsonMpi() = delete;

        //! A private copy constructor (deleted).
        /*!
            �R�s�[�R���X�g���N�^�i�֎~�j
        */
        SimpsonMpi(SimpsonMpi const &) = delete;

        //! A private member function (deleted).
        /*!
            operator=()�̐錾�i�֎~�j
            \param �R�s�[���̃I�u�W�F�N�g
            \return �R�s�[���̃I�u�W�F�N�g
        */
        SimpsonMpi & operator=(SimpsonMpi const &) = delete;

        // #endregion �֎~���ꂽ�R���X�g���N�^�E�����o�֐�
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

        std::int32_t my_rank;                       // �����̃v���Z�X�����N
        // rank�擾
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

        std::int32_t p_num;                         // �v���Z�X��
        // �v���Z�X���擾
        MPI_Comm_size(MPI_COMM_WORLD, &p_num);
                
        std::int32_t localn;                        // �v���Z�b�T����臒l�̕�����
        double localxmax;                           // �v���Z�b�T����臒l�̍ő�l	
        double localxmin;                           // �v���Z�b�T����臒l�̍ŏ��l

        // �v���Z�X����臒l�̎Z�o
        std::tie(localn, localxmin, localxmax) = assign(p_num, my_rank, x1_, x2_);

        // �v���Z�X���ɐϕ�
        auto local_result = simpson(localn, localxmin, localxmax);
        
        double result;                              // �S�̂̐ϕ��l
        std::int32_t dest = 0;
        std::int32_t tag = 0;

        // �e�v���Z�b�T�̐ϕ����ʂ����W
        if (!my_rank) {
            // �v���Z�X0�͑S�Ă̑��v���Z�X���猋�ʂ���M
            fprintf(stdout, "P%d :: x = [%f,%f] : n = %d : s = %f\n", 0,
                localxmin, localxmax, localn, local_result);
            result = local_result;
            for (auto source = 1; source < p_num; source++)
            {
                MPI_Status status;
                // �ϕ����ʂ����W
                MPI_Recv(&local_result, 1, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                
                // �v���Z�X���̌��ʏo��
                std::tie(localn, localxmin, localxmax) = assign(p_num, source, x1_, x2_);
                
                fprintf(stdout, "P%d :: x = [%f,%f] : n = %d : s = %f\n",
                    source, localxmin, localxmax, localn,
                    local_result);

                result += local_result;
            }
        }
        else {
            // �v���Z�X0�ȊO�̓v���Z�X0�֌��ʂ𑗐M
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