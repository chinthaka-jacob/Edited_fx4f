# GitLab Merge Request Instructions

Your changes have been successfully pushed! Here's how to proceed with creating the merge requests.

## Summary

You have made changes across two repositories:

1. **Main Repository** (`fx4f_demo_chinthaka`)
   - Added `simulation.py` configured for John2004 test case
   - Updated submodule pointer to new fx4f branch
   - Branch: `feature/john2004-implementation`
   - URL: https://gitlab.com/chinthaka.jacob/fx4f_demo_chinthaka

2. **Submodule Repository** (`fx4f`)  
   - Added `John2004_1` reference solution implementation
   - New data infrastructure for time-stepping benchmarks
   - Branch: `feature/john2004-reference-solution`
   - URL: https://gitlab.com/fenicsx4flow/fx4f

## Next Steps

### Option 1: Create Merge Request via GitLab Web UI (Recommended)

**Step 1: Create MR in fx4f submodule**

1. Visit: https://gitlab.com/fenicsx4flow/fx4f
2. Click **Merge Requests** (left sidebar)
3. Click **New Merge Request**
4. Set:
   - Source branch: `feature/john2004-reference-solution`
   - Target branch: `main`
5. Click **Compare branches and continue**
6. Fill in the title and description:

```
Title: Add John2004 reference solution with benchmark infrastructure

Description:
## Overview
Implements the manufactured solution from John, Matthies, and Rang (2004) 
for testing time-stepping schemes in the incompressible Navier-Stokes equations.

## Changes
- Add `John2004_1` class: analytical solution on unit square with homogeneous Dirichlet BCs
- Implement time-derivative access for acceleration initialization
- Create data directory structure for time-stepping method comparisons
- Add comprehensive documentation and usage examples
- Include example data file template

## Reference
John, V., Matthies, G., & Rang, J. (2004). *A comparison of time-discretization/linearization 
approaches for the incompressible Navier-Stokes equations.* Computer Methods in Applied Mechanics 
and Engineering, 195(44-47), 5995-6010.

## Testing
The implementation has been tested with:
- Unit square domain (L=1.0)
- Homogeneous Dirichlet boundary conditions
- Generalized-alpha time integration scheme
- Equal-order stabilized finite elements (Q2/Q2)
```

7. Click **Create merge request**

**Step 2: Create MR in main repository**

1. Visit: https://gitlab.com/chinthaka.jacob/fx4f_demo_chinthaka
2. Click **Merge Requests** → **New Merge Request**
3. Set:
   - Source branch: `feature/john2004-implementation`
   - Target branch: `main`
4. Click **Compare branches and continue**
5. Fill in title and description:

```
Title: Configure simulation.py to use John2004 test case

Description:
## Overview
Configures the main simulation framework to solve the John2004 manufactured 
solution with non-periodic Dirichlet boundary conditions.

## Changes
- Import `John2004_1` reference solution
- Replace periodic BCs (with dolfinx_mpc) with Dirichlet BCs on walls
- Update mesh generation for unit square domain
- Simplify nonlinear solver setup
- Add detailed inline comments explaining algorithmic choices
- Update docstring with generalized-alpha methodology

## Technical Details
- Domain: [0,1] × [0,1] unit square
- Boundary conditions: Zero velocity on all walls, zero pressure at corner point
- Time integration: Generalized-alpha scheme (second-order, unconditionally stable)
- Stabilization: SUPG/PSPG/LSIC for equal-order elements
- Finite elements: Q2/Q2 (velocity/pressure)

## Dependencies
- Requires: fx4f!<MR-number> (the John2004 reference solution MR)

## Validation
Successfully runs with default parameters:
- Re = 1, T = 1 s, Δt = 0.05 s
- Mesh: 16×16 elements
- Outputs: velocity and pressure with error norms vs. analytical solution
```

6. Click **Create merge request**

### Option 2: Push and Request Review from Command Line

```bash
# Verify changes are pushed
git -C /app show-ref feature/john2004-implementation
git -C /app/fx4f show-ref feature/john2004-reference-solution

# Get MR URLs for manual creation
echo "Main Repo MR: https://gitlab.com/chinthaka.jacob/fx4f_demo_chinthaka/-/merge_requests/new?merge_request%5Bsource_branch%5D=feature%2Fjohn2004-implementation"
echo "Submodule MR: https://gitlab.com/fenicsx4flow/fx4f/-/merge_requests/new?merge_request%5Bsource_branch%5D=feature%2Fjohn2004-reference-solution"
```

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Main Repository (fx4f_demo_chinthaka)                      │
│  Branch: feature/john2004-implementation                    │
│  - simulation.py (new configuration)                        │
│  - Submodule pointer updated                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
                    (depends on)
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Submodule Repository (fx4f)                                │
│  Branch: feature/john2004-reference-solution               │
│  - John2004_1 class implementation                          │
│  - Data infrastructure                                      │
│  - Documentation                                            │
└─────────────────────────────────────────────────────────────┘
```

## Review Checklist

Before merging, ensure:

- [ ] All tests pass
- [ ] Code follows project style guidelines
- [ ] Documentation is clear and complete
- [ ] Example data file format is documented
- [ ] Submodule MR is approved before main repo MR
- [ ] No breaking changes to existing APIs

## Merge Order

1. Merge fx4f submodule MR first
2. Then merge main repository MR
3. This ensures the submodule pointer is valid

## Verification After Merge

```bash
# Clone and verify
git clone https://gitlab.com/chinthaka.jacob/fx4f_demo_chinthaka.git
cd fx4f_demo_chinthaka
git submodule update --init --recursive

# Test simulation
python simulation.py --Re=1 --T=1 --nx=16 --ny=16
```

## Support

If you encounter any issues:
1. Check git status in both repositories
2. Verify submodule is on correct branch: `cd fx4f && git branch`
3. Update submodule: `git submodule update --recursive`
4. Contact repository maintainers with MR links

---

**Branch Status:**
- Main repo feature branch: ✅ Pushed
- Submodule feature branch: ✅ Already in origin

Ready to create merge requests!
