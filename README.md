MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
Copyright (C) 2013-2019  Manav Bhatia

[![Build Status](https://travis-ci.com/MASTmultiphysics/mast-multiphysics.svg?branch=master)](https://travis-ci.com/MASTmultiphysics/mast-multiphysics)

Documentation for the code is available at [https://mastmultiphysics.github.io](https://mastmultiphysics.github.io).

## Public Release
This code was developed under funding from the Air Force Research Laboratory. 
MAST was cleared for public release via the following case numbers. 

|Date|Case Number|Notes|
|------|------------------|--------|
|08 Nov 2016|88ABW-2016-5689|Initial Release|
|30 Jan 2020|88ABW-2020-0297|Additional Features|



## Submodules
To keep the size of this main MAST repository smaller, a git submodule is used 
to store larger media/assets such as images and animations used for the documentation 
in a separate repo (doc/assets). To build the documentation locally, you must update
the submodule. To do this, simply run the following commands from inside the root
level of this main repository:
```
git submodule init
git submodule update
```
