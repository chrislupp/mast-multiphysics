/*
 * MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
 * Copyright (C) 2013-2020  Manav Bhatia and MAST authors
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

// MAST includes
#include "level_set/level_set_perimeter_output.h"
#include "level_set/level_set_elem_base.h"
#include "level_set/level_set_intersection.h"
#include "level_set/level_set_intersected_elem.h"
#include "base/system_initialization.h"
#include "base/nonlinear_system.h"
#include "base/assembly_base.h"
#include "base/function_base.h"

#include "libmesh/parallel.h"

MAST::LevelSetPerimeter::LevelSetPerimeter():
MAST::OutputAssemblyElemOperations(),
_per                       (0.),
_dper_dp                   (0.),
_heaviside_smooth_delta    (0.1) {
    
}



MAST::LevelSetPerimeter::~LevelSetPerimeter() {
    
    
}



void
MAST::LevelSetPerimeter::init(const MAST::GeomElem& elem) {
    
    libmesh_assert(!_physics_elem);
    libmesh_assert(_system);
    libmesh_assert(_assembly);
    
    _physics_elem = new MAST::LevelSetElementBase(*_system, elem);
}





void
MAST::LevelSetPerimeter::zero_for_analysis() {
    
    _per     = 0.;
    _dper_dp = 0.;
}



void
MAST::LevelSetPerimeter::zero_for_sensitivity() {
    
    // sensitivity does not need the volume data, so we zero both.
    
    _per     = 0.;
    _dper_dp = 0.;
}


Real
MAST::LevelSetPerimeter::output_for_elem() {
    
    libmesh_assert(_physics_elem);
    
    if (this->if_evaluate_for_element(_physics_elem->elem())) {
        
        MAST::LevelSetElementBase&
        e = dynamic_cast<MAST::LevelSetElementBase&>(*_physics_elem);
        
        return e.perimeter(_heaviside_smooth_delta);
    }
    else
    return 0.;
}



Real
MAST::LevelSetPerimeter::output_total() {
    
    Real val = _per;
    
    // sum over all processors since part of the mesh can live on different
    // processors
    _system->system().comm().sum(val);
    
    return val;
}



Real
MAST::LevelSetPerimeter::output_sensitivity_for_elem(const MAST::FunctionBase& p) {
    
    libmesh_assert(_physics_elem);
    const MAST::LevelSetIntersectedElem
    &elem = dynamic_cast<const MAST::LevelSetIntersectedElem&> (_physics_elem->elem());
    
    if (this->if_evaluate_for_element(elem) &&
        elem.if_elem_has_level_set_boundary() &&
        elem.if_subelem_has_side_on_level_set_boundary()) {
        
        MAST::LevelSetElementBase&
        e = dynamic_cast<MAST::LevelSetElementBase&>(*_physics_elem);
        
        return e.perimeter_sensitivity(_heaviside_smooth_delta);
    }
    else
    return 0.;
    
}



Real
MAST::LevelSetPerimeter::output_sensitivity_total(const MAST::FunctionBase& p) {
    
    Real val = _dper_dp;
    
    // sum over all processors since part of the mesh can live on different
    // processors
    _system->system().comm().sum(val);
    
    return val;
}


void
MAST::LevelSetPerimeter::evaluate() {
    
    libmesh_assert(_physics_elem);
    
    if (this->if_evaluate_for_element(_physics_elem->elem())) {
        
        MAST::LevelSetElementBase&
        e = dynamic_cast<MAST::LevelSetElementBase&>(*_physics_elem);
        
        _per += e.perimeter(_heaviside_smooth_delta);
    }
}


void
MAST::LevelSetPerimeter::evaluate_sensitivity(const MAST::FunctionBase& f) {
    
    // nothing to be done here, since this is a topology quantity and
    // the sensitivity is calculated for topology variables
}



void
MAST::LevelSetPerimeter::evaluate_topology_sensitivity(const MAST::FunctionBase& f) {
    
    libmesh_assert(false);
}


void
MAST::LevelSetPerimeter::evaluate_topology_sensitivity(const MAST::FunctionBase& f,
                                                       const MAST::FieldFunction<RealVectorX>& vel) {
    
    // here we ignore the velocity, since the element is able to compute that
    // and provide the sensitivity from level set sensitivity values.
    (void)vel;
    
    libmesh_assert(_physics_elem);
    libmesh_assert(f.is_topology_parameter());
    
    const MAST::LevelSetIntersectedElem
    &elem = dynamic_cast<const MAST::LevelSetIntersectedElem&> (_physics_elem->elem());
    
    // sensitivity only exists at the boundary. So, we proceed with calculation
    // only if this element has an intersection in the interior, or with a side.
    if (this->if_evaluate_for_element(elem)) {

        MAST::LevelSetElementBase&
        e = dynamic_cast<MAST::LevelSetElementBase&>(*_physics_elem);

        _dper_dp += e.perimeter_sensitivity(_heaviside_smooth_delta);
    }
}

