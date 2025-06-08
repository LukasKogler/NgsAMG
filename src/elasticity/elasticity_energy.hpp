#ifndef FILE_ELAST_ENERGY
#define FILE_ELAST_ENERGY

#include <utils_sparseLA.hpp>
#include <utils_denseLA.hpp>

namespace amg
{

template<int ADIM, class T_V_DATA, class T_E_DATA>
class EpsEpsEnergy
{
public:
  using TVD = T_V_DATA;
  using TED = T_E_DATA;

#ifdef ELASTICITY_ROBUST_ECW
  static constexpr bool NEED_ROBUST = true;
#else
  static constexpr bool NEED_ROBUST = false;
#endif

  static constexpr int DIM    = ADIM;
  static constexpr int DISPPV = DIM;
  static constexpr int ROTPV  = ( DIM == 2 ) ? 1 : 3;
  static constexpr int DPV    = ( DIM == 2 ) ? 3 : 6;

  typedef Mat<DPV, DPV, double> TM;

  class TQ
  {
    /**
    * 2D:
    *       1 0 -si * t(1)
    *  Q =  0 1  si * t(0)
    *       0 0     rs
    * 3D:
    *       1 0 0   0    s*t2 -s*t1
    *       0 1 0 -s*t2   0    s*t0
    *  Q =  0 0 1  s*t1 -s*t0   0
    *       0 0 0   rs    0     0
    *       0 0 0   0     rs    0
    *       0 0 0   0     0     rs
    *
    * Where rs = si/sj is the relative r-scaling
    */
  public:
    double rs;
    Vec<DIM, double> tang; // actually s*t

    TQ(TVD const &vdi, TVD const &vdj)
    {
      rs   = vdi.rot_scaling / vdj.rot_scaling;
      tang = vdi.rot_scaling * ( vdj.pos - vdi.pos );
    }

    TQ(double ars, Vec<DIM, double> const &atang)
    {
      rs = ars;
      tang = atang;
    }

    ~TQ() = default;

    static INLINE
    std::tuple<TQ, TQ>
    GetQijQji(TVD const &vdi, TVD const &vdj)
    {
      // double const lambda = vdj.rot_scaling / ( vdi.rot_scaling + vdi.rot_scaling );
      // double const sMid   = sqrt(vdi.rot_scaling * vdj.rot_scaling);

      double const sqrti = sqrt(vdi.rot_scaling);
      double const sqrtj = sqrt(vdj.rot_scaling);

      double const sMid   = sqrti * sqrtj;
      double const lambda = sqrtj / (sqrti + sqrtj);

      Vec<DIM> tij = ( vdj.pos - vdi.pos ); // i -> j
      Vec<DIM> tim =  lambda      * tij;    // i -> mid
      Vec<DIM> tjm = (lambda - 1) * tij;    // j -> mid

      return std::make_tuple(TQ(vdi.rot_scaling / sMid, vdi.rot_scaling * tim),
                             TQ(vdj.rot_scaling / sMid, vdj.rot_scaling * tjm));
    }

    static INLINE
    TQ
    GetQij(TVD const &vdi, TVD const &vdj)
    {
      // double const lambda = vdj.rot_scaling / ( vdi.rot_scaling + vdi.rot_scaling );
      // double const sMid   = sqrt(vdi.rot_scaling * vdj.rot_scaling);
      double const sqrti = sqrt(vdi.rot_scaling);
      double const sqrtj = sqrt(vdj.rot_scaling);

      double const sMid   = sqrti * sqrtj;
      double const lambda = sqrtj / (sqrti + sqrtj);

      Vec<DIM, double> tim = lambda * (vdj.pos - vdi.pos); // i -> mid

      return TQ(vdi.rot_scaling / sMid, vdi.rot_scaling * tim);
    }

    static INLINE
    TQ
    GetQiToj(TVD const &vdi, TVD const &vdj)
    {
      Vec<DIM, double> tij = vdj.pos - vdi.pos;

      return TQ(vdi.rot_scaling / vdj.rot_scaling, vdi.rot_scaling * tij);
    }

    class TT
    {
      /**
      * 2D:
      *     -si * t(1)
      *      si * t(0)
      *
      * 3D:
      *      0    s*t2 -s*t1
      *    -s*t2   0    s*t0
      *     s*t1 -s*t0   0
      */
      public:

      template<class TA>
      static INLINE void
      SetSkT(Vec<DIM, double> const &t,
             TA                     &TT)
      {
        static_assert(Height<TA>() == DISPPV, "SetTT mismatch H A");
        static_assert(Width<TA>()  == ROTPV,  "SetTT mismatch W A");

        if constexpr(DIM == 2)
        {
          TT(0, 0) = -t(1);
          TT(1, 0) =  t(0);
        }
        else
        {
          TT(0,0) = 0;
          TT(1,1) = 0;
          TT(2,2) = 0;
          TT(1,0) = -(TT(0,1) = t[2]);
          TT(0,2) = -(TT(2,0) = t[1]);
          TT(2,1) = -(TT(1,2) = t[0]);
        }

      }


      template<class TA, class TB>
      static INLINE void
      CalcMT(Vec<DIM, double> const &t,
             TA               const &M,
             TB                     &M_T)
      {
        // (NxD) \times (DxR) -> NxR
        constexpr int N  = Height<TA>();

        static_assert(Width<TA>()  == DISPPV, "CalcMT mismatch W A");
        static_assert(Height<TB>() == N,      "CalcMT mismatch H B");
        static_assert(Width<TB>()  == ROTPV,  "CalcMT mismatch W B");

        if constexpr(DIM == 2)
        {
          Iterate<N>([&](auto l)
          {
            M_T(l, 0) = t(0) * M(l, 1) - t(1) * M(l, 0);
          });
        }
        else
        {
          Iterate<N>([&](auto l)
          {
            M_T(l, 0) = t(1) * M(l, 2) - t(2) * M(l, 1);
          });
          Iterate<N>([&](auto l)
          {
            M_T(l, 1) = t(2) * M(l, 0) - t(0) * M(l, 2);
          });
          Iterate<N>([&](auto l)
          {
            M_T(l, 2) = t(0) * M(l, 1) - t(1) * M(l, 0);
          });
        }
      }

      template<class TA, class TB>
      static INLINE void
      CalcTTM(Vec<DIM, double> const &t,
              TA               const &M,
              TB                     &TT_M)
      {
        // (RxD) \times (DxN) -> RxN
        constexpr int N  = Width<TA>();

        static_assert(Height<TA>() == DISPPV, "CalcTTM mismatch H A");
        static_assert(Height<TB>() == ROTPV,  "CalcTTM mismatch H B");
        static_assert(Width<TB>()  == N,      "CalcTTM mismatch W B");

        if constexpr(DIM == 2)
        {
          Iterate<N>([&](auto l)
          {
            TT_M(0, l) = t(0) * M(1, l) - t(1) * M(0, l);
          });
        }
        else
        {
          Iterate<N>([&](auto l)
          {
            TT_M(0, l) = t(1) * M(2, l) - t(2) * M(1, l);
          });
          Iterate<N>([&](auto l)
          {
            TT_M(1, l) = t(2) * M(0, l) - t(0) * M(2, l);
          });
          Iterate<N>([&](auto l)
          {
            TT_M(2, l) = t(0) * M(1, l) - t(1) * M(0, l);
          });
        }
      }

      template<class TA, class TB>
      static INLINE void
      CalcTTMSymm(Vec<DIM, double> const &t,
                  TA               const &A_T,
                  TB                     &TT_A_T)
      {
        static_assert(Height<TA>() == DISPPV, "CalcTTR mismatch - H A");
        static_assert(Width<TA>()  == ROTPV,  "CalcTTR mismatch - H A");
        static_assert(Height<TB>() == ROTPV,  "CalcTTR mismatch - H B");
        static_assert(Width<TB>()  == ROTPV,  "CalcTTR mismatch - W B");

        // (RxD) \times (DxR) -> RxR
        if constexpr(DIM == 2)
        {
          TT_A_T(0, 0) = t(0) * A_T(1, 0) - t(1) * A_T(0, 0);
        }
        else
        {
          // first row - calc 0,1,2
          Iterate<3>([&](auto l)
          {
            TT_A_T(0, l) = t(1) * A_T(2, l) - t(2) * A_T(1, l);
          });

          // second row - calc 1,2
          TT_A_T(1, 0) = TT_A_T(0, 1);

          Iterate<2>([&](auto ll)
          {
            constexpr int l = ll + 1;

            TT_A_T(1, l) = t(2) * A_T(0, l) - t(0) * A_T(2, l);
          });

          // third row - calc 3
          TT_A_T(2, 0) = TT_A_T(0, 2);
          TT_A_T(2, 1) = TT_A_T(1, 2);
          TT_A_T(2, 2)   = t(0) * A_T(1, 2) - t(1) * A_T(0, 2);
        }
      }


      template<class T_A, class T_A_T, class T_TT_A_T>
      static INLINE void
      CalcTTAT(Vec<DIM, double> const &t,
               T_A              const &A,
               T_A_T                  &A_T,
               T_TT_A_T               &TT_A_T)
      {
        static_assert(Height<T_A>()      == DISPPV, "CalcTTAT mismatch - H A");
        static_assert(Width<T_A>()       == DISPPV, "CalcTTAT mismatch - W A");
        static_assert(Height<T_A_T>()    == DISPPV, "CalcTTAT mismatch - H B");
        static_assert(Width<T_A_T>()     == ROTPV,  "CalcTTAT mismatch - W B");
        static_assert(Height<T_TT_A_T>() == ROTPV,  "CalcTTAT mismatch - H C");
        static_assert(Width<T_TT_A_T>()  == ROTPV,  "CalcTTAT mismatch - W C");

        CalcMT(t, A, A_T);
        CalcTTMSymm(t, A_T, TT_A_T);
      }


      template<class TA, class TB>
      static INLINE void
      CalcTM(Vec<DIM, double> const &t,
             TA               const &M,
             TB                     &T_M)
      {
        // DxR \times RxN -> DxN
        constexpr int N  = Width<TB>();

        static_assert(Height<TA>() == ROTPV,  "CalcTM mismatch - H A");
        static_assert(Width<TA>()  == N,      "CalcTM mismatch - W A");
        static_assert(Height<TB>() == DISPPV, "CalcTM mismatch - H B");

        if constexpr(DIM == 2)
        {
          /**
           *   -si * t(1)
           *    si * t(0)  \times M
           */
          Iterate<N>([&](auto l)
          {
            T_M(0, l) = -t(1) * M(0, l);
            T_M(1, l) =  t(0) * M(0, l);
          });
        }
        else
        {
         /*
          *    0    s*t2 -s*t1
          *  -s*t2   0    s*t0  \times  M
          *   s*t1 -s*t0   0
          */

          Iterate<N>([&](auto l)
          {
            T_M(0, l) = t(1) * M(2, l) - t(2) * M(1, l);
          });

          Iterate<N>([&](auto l)
          {
            T_M(1, l) = t(2) * M(0, l) - t(0) * M(2, l);
          });

          Iterate<N>([&](auto l)
          {
            T_M(2, l) = t(0) * M(1, l) - t(1) * M(0, l);
          });
        }

      }

      template<class TA, class TB>
      static INLINE void
      CalcMTT(Vec<DIM, double> const &t,
              TA               const &M,
              TB                     &M_TT)
      {
        // NxR \times RxD -> NxD
        constexpr int N = Height<TA>();

        static_assert(Width<TA>()  == ROTPV,  "CalcMTT mismatch - W A");
        static_assert(Height<TB>() == N,      "CalcMTT mismatch - H B");
        static_assert(Width<TB>()  == DISPPV, "CalcMTT mismatch - W B");

        if constexpr(DIM == 2)
        {
          // M \times ( -t(1) | t(0) )
          Iterate<N>([&](auto l)
          {
            M_TT(l, 0) = -t(1) * M(l, 0);
            M_TT(l, 1) =  t(0) * M(l, 0);
          });
        }
        else
        {
          /*
           *              0   -s*t2  s*t1
           * M  \times   s*t2   0   -s*t0
           *             -s*t1  s*t0   0
           */
          Iterate<N>([&](auto l)
          {
            M_TT(l, 0) = t(2) * M(l, 1) - t(1) * M(l, 2);
          });

          Iterate<N>([&](auto l)
          {
            M_TT(l, 1) = t(0) * M(l, 2) - t(2) * M(l, 0);
          });

          Iterate<N>([&](auto l)
          {
            M_TT(l, 2) = t(1) * M(l, 0) - t(0) * M(l, 1);
          });
        }
      }

      template<class T_T_M, class T_T_M_TT>
      static INLINE void
      CalcMTTSymm(Vec<DIM, double> const &t,
                  T_T_M                  &T_M,
                  T_T_M_TT               &T_M_TT)
      {
        // DxR \times RxR \times RxD -> DxD
        static_assert(Height<T_T_M>()    == DISPPV, "CalcMTTSymm mismatch - H A");
        static_assert(Width<T_T_M>()     == ROTPV,  "CalcMTTSymm mismatch - W A");
        static_assert(Height<T_T_M_TT>() == DISPPV, "CalcMTTSymm mismatch - H B");
        static_assert(Width<T_T_M_TT>()  == DISPPV, "CalcMTTSymm mismatch - W B");

        if constexpr(DIM == 2)
        {
          // T_M \times ( -t(1) | t(0) )
          T_M_TT(0, 0) = -t(1) * T_M(0, 0);
          T_M_TT(0, 1) =  t(0) * T_M(0, 0);
          T_M_TT(1, 0) = T_M_TT(0, 1);
          T_M_TT(1, 1) =  t(0) * T_M(1, 0);
        }
        else
        {
          /*
           *              0   -s*t2  s*t1
           * M  \times   s*t2   0   -s*t0
           *             -s*t1  s*t0   0
           */
          Iterate<3>([&](auto l)
          {
            T_M_TT(l, 0) = t(2) * T_M(l, 1) - t(1) * T_M(l, 2);
          });

          // second column
          T_M_TT(0, 1) = T_M_TT(1, 0);
          T_M_TT(1, 1) = t(0) * T_M(1, 2) - t(2) * T_M(1, 0);
          T_M_TT(2, 1) = t(0) * T_M(2, 2) - t(2) * T_M(2, 0);

          // third column
          T_M_TT(0, 2) = T_M_TT(2, 0);
          T_M_TT(1, 2) = T_M_TT(2, 1);
          T_M_TT(0, 2) = t(1) * T_M(1, 0) - t(0) * T_M(2, 1);
        }
      }

      template<class TYPE_R, class TYPE_T_R, class TYPE_T_R_TT>
      static INLINE void
      CalcTRTT(Vec<DIM, double> const &t,
               TYPE_R           const &R,
               TYPE_T_R               &T_R,
               TYPE_T_R_TT            &T_R_TT)
      {
        // DISP x ROT \times ROT x ROT \times ROT x DISP
        static_assert(Height<TYPE_R>()      == ROTPV,  "CalcTRTT mismatch - H A");
        static_assert(Width<TYPE_R>()       == ROTPV,  "CalcTRTT mismatch - W A");
        static_assert(Height<TYPE_T_R>()    == DISPPV, "CalcTRTT mismatch - H B");
        static_assert(Width<TYPE_T_R>()     == ROTPV,  "CalcTRTT mismatch - W B");
        static_assert(Height<TYPE_T_R_TT>() == DISPPV, "CalcTRTT mismatch - H C");
        static_assert(Width<TYPE_T_R_TT>()  == DISPPV, "CalcTRTT mismatch - W C");

        CalcTM(t, R, T_R);
        CalcMTTSymm(t, T_R, T_R_TT);
      }

    };


    template<class TMU>
    INLINE void
    SetQ(double const &alpha, TMU &E) const
    {
      /**
       *  |I  T |
       *  |0 r*I|
       */
      // (I, 0)^T
      Iterate<DPV>([&](auto ii)
      {
        Iterate<DISPPV>([&](auto jj)
        {
          if constexpr(ii.value == jj.value)
          {
            E(ii.value, jj.value) = alpha; // 1.0;
          }
          else
          {
            E(ii.value, jj.value) = 0.0;
          }
        });
      });

      // T
      FlatMat<0, DISPPV, DISPPV, ROTPV,  TMU> T(E); // A and C
      TT::SetSkT(tang, T);

      // r*I
      FlatMat<DISPPV, ROTPV, DISPPV, ROTPV,  TMU> RI(E); // A and C
      Iterate<ROTPV>([&](auto ii)
      {
        Iterate<ROTPV>([&](auto jj)
        {
          if constexpr(ii == jj)
          {
            RI(ii.value, jj.value) = alpha * rs; // rs;
          }
          else
          {
            RI(ii.value, jj.value) = 0.0;
          }
        });
      });
    }

    template<class TSCAL>
    INLINE void
    SetQ(Mat<DPV, DPV, TSCAL> &E) const
    {
      SetQ(1.0, E);
    }

    template<class TSCAL>
    INLINE void
    MQ(Mat<DPV, DPV, TSCAL> &E) const
    {
      /**
       *  |A B|  |I  T |  ->      |A   r*B| + | 0 A*T |
       *  |C D|  |0 r*I|  ->      |C   r*D| + | 0 C*T |
       */
      using TMU = Mat<DPV, DPV, TSCAL>;

      FlatMat<0, DPV, 0,      DISPPV, TMU> AC(E); // A and C
      FlatMat<0, DPV, DISPPV, ROTPV,  TMU> BD(E); // A and C

      if (rs != 1.0)
      {
        BD *= rs;
      }

      Mat<DPV, ROTPV, TSCAL> AC_T;
      TT::CalcMT(tang, AC, AC_T);

      BD += AC_T;
    }

    template<class TSCAL>
    INLINE void
    MQ(TSCAL const &scal, TM const &E, TM &E_Q) const
    {
      E_Q = scal * E;
      MQ(E_Q);
    }

    template<class TSCAL>
    INLINE TM
    GetMQ(TSCAL const &scal, TM const &E) const
    {
      TM E_Q;
      MQ(scal, E, E_Q);
      return E_Q;
    }

    template<class TSCAL>
    INLINE void
    QTM(Mat<DPV, DPV, TSCAL> &E) const
    {
      /**
       *  | I   0 |  |A  B|    ->  | A   B  | + |  0     0 |
       *  | TT r*I|  |C  D|    ->  |r*C  r*D| + |TT*A  TT*B|
       */
      using TMU = Mat<DPV, DPV, TSCAL>;

      FlatMat<0,      DISPPV, 0, DPV, TMU> AB(E);
      FlatMat<DISPPV, ROTPV,  0, DPV, TMU> CD(E);

      if (rs != 1.0)
      {
        CD *= rs;
      }

      Mat<ROTPV, DPV, TSCAL> TT_AB;
      TT::CalcTTM(tang, AB, TT_AB);

      CD += TT_AB;
    }

    template<class TSCAL, class TM>
    INLINE void
    QTM(TSCAL const &scal, TM const &E, TM &QT_E) const
    {
      QT_E = scal * E;
      QTM(QT_E);
    }

    template<class TSCAL, class TM>
    INLINE TM
    GetQTM(TSCAL const &scal, TM const &E) const
    {
      TM QT_E;
      QTM(scal, E, QT_E);
      return QT_E;
    }

    template<class TSCAL>
    INLINE void
    QTMQ(Mat<DPV,DPV,TSCAL> &E) const
    {
      /**
       *  |I   0 | |A  B| |I  T |  ->          A                    A*T + r*B
       *  |TT r*I| |BT C| |0 r*I|  ->     TT*A + r*BT    r*r*C + r*TT*B + r*BT*T + TT*A*T
       */
      typedef Mat<DPV,DPV,TSCAL> TMU;

      FlatMat<0,      DISPPV, 0,      DISPPV, TMU> A(E);
      FlatMat<0,      DISPPV, DISPPV, ROTPV,  TMU> B(E);
      FlatMat<DISPPV, ROTPV,  0     , DISPPV, TMU> BT(E);
      FlatMat<DISPPV, ROTPV,  DISPPV, ROTPV,  TMU> C(E);

      Mat<DISPPV, ROTPV, TSCAL> A_T;
      Mat<ROTPV,  ROTPV, TSCAL> r_TT_B;
      Mat<ROTPV,  ROTPV, TSCAL> TT_A_T;

      TT::CalcTTAT(tang, A, A_T, TT_A_T);
      TT::CalcTTM(tang, B, r_TT_B);

      if (rs != 1.0)
      {
        C      *= rs * rs;
        B      *= rs;
        r_TT_B *= rs;
      }

      B += A_T;
      BT = Trans(B);

      C += TT_A_T + r_TT_B + Trans(r_TT_B);
    }

    template<class TSCAL, class TM>
    INLINE void
    QTMQ(TSCAL const &scal, TM const &E, TM &QT_E_Q) const
    {
      QT_E_Q = scal * E;
      QTMQ(QT_E_Q);
    }

    template<class TSCAL, class TM>
    INLINE TM
    GetQTMQ(TSCAL const &scal, TM const &E) const
    {
      TM QT_E_Q;
      QTMQ(scal, E, QT_E_Q);
      return QT_E_Q;
    }

    template<class TSCAL>
    INLINE void
    QMQT(Mat<DPV,DPV,TSCAL> &E) const
    {
      /**
       *  I     T    A  B     I  0    ->    A + B_TT + T_BT + T_C_TT   r(B + TC)
       *  0    r*I   BT C    TT  r*I  ->        r (BT + C_TT)            r^2*C
       */
      typedef Mat<DPV,DPV,TSCAL> TMU;

      FlatMat<0,      DISPPV, 0,      DISPPV, TMU> A(E);
      FlatMat<0,      DISPPV, DISPPV, ROTPV,  TMU> B(E);
      FlatMat<DISPPV, ROTPV,  0     , DISPPV, TMU> BT(E);
      FlatMat<DISPPV, ROTPV,  DISPPV, ROTPV,  TMU> C(E);

      Mat<DISPPV, ROTPV,  TSCAL>  T_C;
      Mat<DISPPV, DISPPV, TSCAL>  T_C_TT;
      Mat<DISPPV, DISPPV, TSCAL>  B_TT;

      TT::CalcTRTT(tang, C, T_C, T_C_TT);
      TT::CalcMTT(tang, B, B_TT);

      B += T_C;

      if ( rs != 1.0 )
      {
        B *= rs;
        C *= rs * rs;
      }

      BT = Trans(B);

      A += B_TT + Trans(B_TT) + T_C_TT;
    }

    template<class TSCAL, class TM>
    INLINE void
    QMQT(TSCAL const &scal, TM const &E, TM &Q_E_QT) const
    {
      Q_E_QT = scal * E;
      QMQT(Q_E_QT);
    }

    template<class TSCAL, class TM>
    INLINE TM
    GetQMQT(TSCAL const &scal, TM const &E) const
    {
      TM Q_E_QT;
      QMQT(scal, E, Q_E_QT);
      return Q_E_QT;
    }
  };

  template<class TSCAL = double>
  static INLINE TSCAL
  GetApproxWeight (TED const &ed)
  {
    return TSCAL(CalcAvgTrace(ed));
  }

  template<class TSCAL = double>
  static INLINE TSCAL
  GetApproxVWeight (TVD const &vd)
  {
    return TSCAL(CalcAvgTrace(vd.wt));
  }


  template<class TSCAL = double>
  static INLINE Mat<DPV,DPV,TSCAL> GetEMatrix (TED const &ed)
  {
    Mat<DPV,DPV,TSCAL> m = ed;
    return m;
  }

  // static INLINE TM const &GetVMatrix (const TVD & vd) { return vd.wt; }

  template<class TSCAL = double>
  static INLINE Mat<DPV,DPV,TSCAL> GetVMatrix (TVD const &vd)
  {
    Mat<DPV,DPV,TSCAL> m = vd.wt;
    return m;
  }

  template<class TMU>
  static INLINE void
  SetEMatrix (TED & ed, TMU const &m)
  {
    if constexpr(std::is_same<TMU, float>::value ||
                 std::is_same<TMU, double>::value)
    {
      SetScalIdentity(m, ed);
    }
    else
    {
      // static_assert(std::is_same<TM, TMU>::value, "Elasticity SetEMatrix");
      ed = m;
    }
  }

  template<class TMU>
  static INLINE void
  SetVMatrix (TVD & vd, TMU const &m)
  {
    if constexpr(std::is_same<TMU, float>::value ||
                 std::is_same<TMU, double>::value)
    {
      SetScalIdentity(m, vd.wt);
    }
    else
    {
      // static_assert(std::is_same<TM, TMU>::value, "Elasticity SetVMatrix");
      static_assert(Height<TM>() == Height<TMU>(), "Elasticity SetVMatrix");

      vd.wt = m;
    }
  }

  template<bool BOTH_ROWS, class TLAM>
  static INLINE void
  CalcRMBlockImpl (const TED & ed,
                   const TVD & vdi,
                   const TVD & vdj,
                   TLAM lam)
  {
    auto const &E = GetEMatrix(ed);

    auto [Qij, Qji] = TQ::GetQijQji(vdi, vdj);

    TM QijT_E; Qij.QTM(1.0, E, QijT_E);

    TM QijT_E_Qij; Qij.MQ(1.0, QijT_E, QijT_E_Qij);
    lam(0, 0, QijT_E_Qij);

    TM QijT_E_Qji; Qji.MQ(-1.0, QijT_E, QijT_E_Qji);
    lam(0, 1, QijT_E_Qji);

    if constexpr(BOTH_ROWS)
    {
      TM QjiT_E_Qij = Trans(QijT_E_Qji);
      lam(1, 0, QjiT_E_Qij);

      TM QjiT_E_Qji; Qji.QTMQ(1.0, E, QjiT_E_Qji);
      lam(1, 1, QjiT_E_Qji);
    }
  }

  template<class TLAM>
  static INLINE void
  CalcRMBlock(const TED & ed,
              const TVD & vdi,
              const TVD & vdj,
              TLAM lam)
  {
    CalcRMBlockImpl<true>(ed, vdi, vdj, lam);
  }

  template<class TLAM>
  static INLINE void
  CalcRMBlockRow(const TED & ed,
                 const TVD & vdi,
                 const TVD & vdj,
                 TLAM lam)
  {
    CalcRMBlockImpl<false>(ed, vdi, vdj, lam);
  }


  static INLINE std::tuple<TQ, TQ>
  GetQijQji  (const TVD & di, const TVD & dj)
  {
    return TQ::GetQijQji(di, dj);
  }

  static INLINE TQ
  GetQij  (const TVD & di, const TVD & dj)
  {
    return TQ::GetQij(di, dj);
  }

  static INLINE TQ
  GetQiToj(TVD const &vdi, TVD const &vdj)
  {
    return TQ::GetQiToj(vdi, vdj);
  }

  static INLINE void CalcQ  (const Vec<DIM> & t, TM & Q, double si, double sj);
  static INLINE void ModQ  (const Vec<DIM> & t, TM & Q, double si, double sj);
  static INLINE void CalcQij (const TVD & di, const TVD & dj, TM & Qij);
  static INLINE void ModQij (const TVD & di, const TVD & dj, TM & Qij);
  static INLINE void CalcQHh (const TVD & dH, const TVD & dh, TM & QHh, double glob_scale = 1.0);
  static INLINE void ModQHh (const TVD & dH, const TVD & dh, TM & QHh, double glob_scale = 1.0);
  static INLINE void CalcQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
  static INLINE void CalcInvQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
  static INLINE void CalcK (const TVD& di, const TVD& dj, FlatVector<TM> K);
  static INLINE void ModQs  (const TVD & di, const TVD & dj, TM & Qij, TM & Qji);
  static INLINE TVD CalcMPData (const TVD & da, const TVD & db);
  static INLINE TVD CalcMPDataWW (const TVD & da, const TVD & db); // with weights

  static INLINE void CalcRMBlock (FlatMatrix<TM> mat, const TED & ed, const TVD & vdi, const TVD & vdj);
  static INLINE void CalcRMBlock2 (FlatMatrix<TM> mat, const TM & ed, const TVD & vdi, const TVD & vdj);

  static INLINE void QtMQ (const TM & Qij, TM & M)
  {
    /** I         A  B      I Q
        QT s*I    BT D       s*I **/
    static Mat<DISPPV, ROTPV, double> AQ;
    static Mat<ROTPV, ROTPV, double> M22X;
    const double s = Qij(DISPPV, DISPPV);
    auto Q = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
    auto A = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    auto B = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    auto BT = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
    auto D = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
    AQ = A * Q;
    D *= s*s;
    D += Trans(Q) * (AQ + (s*B)) + (s*BT) * Q;
    B = s * B + AQ; // should be OK
    // B *= s;
    // B += AQ;
    BT = Trans(B);
  }

  static INLINE void AddQtMQ (double val, TM & aA, const TM & Qij, const TM & M)
  {
    /** I         A  B     I Q
        QT s*I    BT D      s*I **/
    static Mat<DISPPV, ROTPV, double> AQpB;
    const double s = Qij(DISPPV, DISPPV);
    auto Q = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
    auto A = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    auto B = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    auto BT = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
    auto D = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
    AQpB = A * Q + (s*B);
    MakeFlatMat<0, DISPPV, 0, DISPPV>(aA) += val * A;
    MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(aA) += val * AQpB;
    MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(aA) += val * Trans(AQpB);
    MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(aA) += val * (Trans(Q) * AQpB + (s*BT) * Q + (sqr(s)*D) );
  }

  static INLINE TM CalcQtMQ (const TM & Qij, const TM & M)
  {
    TM MQM = M;
    QtMQ(Qij, MQM);
    return MQM;
  }

  static INLINE void SetQtMQ (double val, TM & aA, const TM & Qij, const TM & M)
  {
    /** I       A  B    I Q
        QT I    BT D      I **/
    static Mat<DISPPV, ROTPV, double> AQpB;
    const double s = Qij(DISPPV, DISPPV);
    auto Q = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Qij);
    auto A = MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    auto B = MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    auto BT = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
    auto D = MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M);
    AQpB = A * Q + (s*B);
    MakeFlatMat<0, DISPPV, 0, DISPPV>(aA) = val * A;
    MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(aA) = val * AQpB;
    MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(aA) = val * Trans(AQpB);
    MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(aA) = val * (Trans(Q) * AQpB + BT * Q + (sqr(s)*D) );
  }


  /** Fake harmonic mean - do not use this for stable coarsening ! **/
  static INLINE TM HMean (const TM & A, const TM & B)
  {
    double tra = calc_trace(A), trb = calc_trace(B);
    double tr = (2.0*tra*trb)/(tra+trb);
    return tr * 0.5 * (A/tra + B/trb);
  }


  /** Fake geometric mean - do not use this for stable coarsening ! **/
  static INLINE TM GMean (const TM & A, const TM & B)
  {
    double tra = calc_trace(A), trb = calc_trace(B);
    double tr = sqrt(tra*trb);
    return tr * 0.5 * (A/tra + B/trb);
  }

  /** Actually the real algebraic mean **/
  static INLINE TM AMean (const TM & A, const TM & B)
  {
    return 0.5 * (A + B);
  }

  static INLINE void CalcMQ (double scal, const TM & M, const TM & Q, TM & out)
  {
    /** A  B   I  Q   =  A    AQ+s*B
        BT C   0 s*I  =  BT  BTQ+s*C **/
    static Mat<DISPPV, ROTPV, double> AQ;
    static Mat<ROTPV, ROTPV, double> BTQ;
    const double s = Q(DISPPV, DISPPV);
    BTQ = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
    AQ = MakeFlatMat<0, DISPPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
    MakeFlatMat<0, DISPPV, 0, DISPPV>(out) = scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) = scal * ( (s*MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M)) + AQ );
    MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) = scal * MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
    MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) = scal * ( (s*MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M)) + BTQ);
  }

  static INLINE void AddMQ (double scal, const TM & M, const TM & Q, TM & out)
  {
    /** A  B   I  Q   =  A   AQ+s*B
        BT C   0 s*I  =  BT BTQ+s*C **/
    static Mat<DISPPV, ROTPV, double> AQ;
    static Mat<ROTPV, ROTPV, double> BTQ;
    const double s = Q(DISPPV, DISPPV);
    BTQ = MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
    AQ = MakeFlatMat<0, DISPPV, 0, DISPPV>(M) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q);
    MakeFlatMat<0, DISPPV, 0, DISPPV>(out) += scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) += scal * ( (s*MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M)) + AQ );
    MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) += scal * MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M);
    MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) += scal * ( (s*MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M)) + BTQ);
  }

  static INLINE void CalcQTM (double scal, const TM & Q, const TM & M, TM & out)
  {
    /** I   0    A  B   =     A        B
        QT s*I   BT C   =  QTA+s*BT QTB+s*C **/
    static Mat<DISPPV, ROTPV, double> QTA;
    static Mat<ROTPV, ROTPV, double> QTB;
    const double s = Q(DISPPV, DISPPV);
    QTA = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    QTB = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    MakeFlatMat<0, DISPPV, 0, DISPPV>(out) = scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) = scal * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) = scal * ( (s*MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M)) + QTA );
    MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) = scal * ( (s*MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M)) + QTB);
  }

  static INLINE void AddQTM (double scal, const TM & Q, const TM & M, TM & out)
  {
    /** I  0   A  B   =    A      B
        QT I   BT C   =  QTA+BT QTB+C **/
    static Mat<DISPPV, ROTPV, double> QTA;
    static Mat<ROTPV, ROTPV, double> QTB;
    const double s = Q(DISPPV, DISPPV);
    QTA = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    QTB = Trans(MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(Q)) * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    MakeFlatMat<0, DISPPV, 0, DISPPV>(out) += scal * MakeFlatMat<0, DISPPV, 0, DISPPV>(M);
    MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(out) += scal * MakeFlatMat<0, DISPPV, DISPPV, ROTPV>(M);
    MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(out) += scal * ( (s*MakeFlatMat<DISPPV, ROTPV, 0, DISPPV>(M)) + QTA );
    MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(out) += scal * ( (s*MakeFlatMat<DISPPV, ROTPV, DISPPV, ROTPV>(M)) + QTB);
  }
}; // class EpsEpsEnergy

} // namespace amg

#endif // FILE_ELAST_ENERGY