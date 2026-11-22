The DORBAN Code
====================
.. automodule:: dorban
   :members:
   :no-undoc-members:
   :show-inheritance:




Code Flow
    1. The user defines a :class:`Core <dorban.system.Core>` by supplying a
       sequence of :class:`CrossSectionData <dorban.materials.CrossSectionData>`,
       the number of energy groups, the :class:`Geometry <dorban.geometry.geometry.Geometry>`
       and a :class:`CurrentCalculator <dorban.current_calculator.CurrentCalculator>`.
       The Geomerty also contains the :class:`BoundaryConditions <dorban.geometry.boundary_conditions.Boundary>`.
    2. The user defines a :class:`Settings <dorban.settings.Settings>` object by supplying a
       split for the geometry and parameters for the solver, such as the various tolerances
       and limitations on the number of iterations
    3. The user chooses which problem to solve, for example a k problem or an adjoint
       flux problem.
    4. The code refines the mesh of the core :class:`Core <dorban.system.Core>` defined by the user using the split data of the :class:`Settings <dorban.settings.Settings>` object.
    5. The code constructs 3 sparse matrices in :class:`csr` format, the absorption matrix, the diffusion matrix and the fission matrix.
    6. The code passes to the eigenvalue solver a combination of these matrices which represents the problem the user wants to solve.
    7. The eigenvalue and eigenvector are computed by the solver.
    8. The eigenvector is condensed back to the mesh of the original :class:`Core <dorban.system.Core>` defined by the user.
    9. The eigenvalue and the condensed eigenvector are returne


.. toctree::
   :hidden:
   :maxdepth: 2

   dorban.system
   dorban.geometry
   dorban.boundary_conditions
   dorban.materials
   dorban.cross_section_operators
   dorban.current_calculator
   dorban.rdfs
   dorban.finite_differences
   dorban.solve_equation
   dorban.eigenvalue_solvers
   dorban.settings
   dorban.utils
   dorban.nem
   dorban.examples
