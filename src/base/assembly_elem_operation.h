/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2018  Manav Bhatia
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#ifndef __mast_assembly_elem_operation_h__
#define __mast_assembly_elem_operation_h__


// libMesh includes
#include "libmesh/elem.h"

namespace MAST {
    
    // Forward declerations
    class FEBase;
    class ElementBase;
    class AssemblyBase;
    class LocalElemFE;
    
    class AssemblyElemOperations {
        
    public:
        AssemblyElemOperations();
        
        virtual ~AssemblyElemOperations();
        
        
        /*!
         *   sets the assembly object
         */
        void set_assembly(MAST::AssemblyBase& assembly);

        /*!
         *   @returns a reference to the assembly object
         */
        MAST::AssemblyBase& get_assembly();
        
        /*!
         *   clears the assembly object
         */
        void clear_assembly();

        
        /*!
         *   @returns a smart-pointer to a newly created element for
         *   calculation of element quantities.
         */
        virtual std::unique_ptr<MAST::ElementBase>
        build_elem(const libMesh::Elem& elem) = 0;

        
        /*!
         *   some simulations frequently deal with 1D/2D elements in 3D space,
         *   which requires use of MAST::LocalElemFE.
         */
        virtual bool
        if_use_local_elem() const = 0;

        
        /*!
         *   sets additional data for local elem FE.
         */
        virtual void
        set_local_fe_data(const libMesh::Elem& e,
                          MAST::LocalElemFE& fe) const = 0;

    protected:


        MAST::AssemblyBase *_assembly;
    };
}


#endif // __mast_assembly_elem_operation_h__

