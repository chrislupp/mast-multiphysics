
#ifndef __mast__first_order_newmark_transient_solver__
#define __mast__first_order_newmark_transient_solver__

// MAST includes
#include "solver/transient_solver_base.h"


namespace MAST {
    
    
    /*!
     *    This class implements the Newmark solver for solution of a
     *    first-order ODE.
     *
     *    The residual here is modeled as
     *    \f[ r = beta*dt(f_m + f_x )= 0 \f]
     *    where, (for example)
     *    \f{eqnarray*}{
     *      f_m & = &  int_Omega phi u_dot   \mbox{ [typical mass vector in conduction, for example]}\\
     *      f_x & = &  int_Omega phi_i u_i - int_Gamma phi q_n \mbox{ [typical conductance and heat flux combination, for example]}
     *    \f}
     *
     *    This method assumes
     *    \f{eqnarray*}{
     *     x     &=& x0 + (1-beta) dt x0_dot + beta dt x_dot \\
     *    \mbox{or, } x_dot &=& (x-x0)/beta/dt - (1-beta)/beta x0_dot
     *    }
     *    Note that the residual expression multiplies the expression by beta*dt
     *    for convenience in the following expressions
     *
     *    Both f_m and f_x can be functions of x_dot and x. Then, the
     *    Jacobian is
     *    \f{eqnarray*}{
     *    dr/dx &=& beta*dt [df_m/dx + df_x/dx +\\
     *       &&          df_m/dx_dot dx_dot/dx + df_x/dx_dot dx_dot/dx]\\
     *       &=& beta*dt [(df_m/dx + df_x/dx) +\\
     *       &&           (df_m/dx_dot + df_x/dx_dot) (1/beta/dt)]\\
     *       &=& beta*dt (df_m/dx + df_x/dx) +\\
     *       &&          (df_m/dx_dot + df_x/dx_dot)
     *    }
     *   Note that this form of equations makes it a good candidate for
     *   use as implicit solver, ie, for a nonzero beta.
     */
    class FirstOrderNewmarkTransientSolver:
    public MAST::TransientSolverBase {
    public:
        FirstOrderNewmarkTransientSolver();
        
        virtual ~FirstOrderNewmarkTransientSolver();
        
        /*!
         *    \f$ \beta \f$ parameter used by this solver.
         */
        Real beta;
        
        /*!
         *    @returns the highest order time derivative that the solver
         *    will handle
         */
        virtual int ode_order() const {
            return 1;
        }

        
        /*!
         *   solves the current time step for solution and velocity
         */
        virtual void solve();
        
        /*!
         *   advances the time step and copies the current solution to old
         *   solution, and so on.
         */
        virtual void advance_time_step();
        
    protected:
        
        /*!
         *    @returns the number of iterations for which solution and velocity
         *    are to be stored.
         */
        virtual unsigned int _n_iters_to_store() const {
            return 2;
        }
        
        /*!
         *    provides the element with the transient data for calculations
         */
        virtual void _set_element_data(std::vector<libMesh::dof_id_type>& dof_indices,
                                       MAST::ElementBase& elem);

        /*!
         *    update the transient velocity based on the current solution
         */
        virtual void _update_velocity(libMesh::NumericVector<Real>& vec);
        
        /*!
         *    update the transient acceleration based on the current solution
         */
        virtual void _update_acceleration(libMesh::NumericVector<Real>& vec) {
            // should not get here for first order ode
            libmesh_error();
        }

        /*!
         *   performs the element calculations over \par elem, and returns
         *   the element vector and matrix quantities in \par mat and
         *   \par vec, respectively. \par if_jac tells the method to also
         *   assemble the Jacobian, in addition to the residual vector.
         */
        virtual void
        _elem_calculations(MAST::ElementBase& elem,
                           const std::vector<libMesh::dof_id_type>& dof_indices,
                           bool if_jac,
                           RealVectorX& vec,
                           RealMatrixX& mat);
        
        /*!
         *   performs the element sensitivity calculations over \par elem,
         *   and returns the element residual sensitivity in \par vec .
         */
        virtual void
        _elem_sensitivity_calculations(MAST::ElementBase& elem,
                                       const std::vector<libMesh::dof_id_type>& dof_indices,
                                       RealVectorX& vec);
    };
    
}

#endif // __mast__first_order_newmark_transient_solver__