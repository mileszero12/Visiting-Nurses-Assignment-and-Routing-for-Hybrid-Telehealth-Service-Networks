#ifndef CODE_SOLVER_H
#define CODE_SOLVER_H

#include "gurobi_c.h"
#include "gurobi_c++.h"
#include <ctime>

/* Class for solver */
class Solver{
public:
    // get basic info (row, col, status, opt)
    int SolverGetRowNum(int *const pvalue);                               // get row number
    int SolverGetColNum(int *const pvalue);                               // get column number
    int SolverGetStatus(int *const pvalue);                               // get current status
    int SolverOptimize();                                                 // solve problem
    int SolverUpdate();                                                   // model update

    // free model and environment
    void SolverFreeModel();                                               // free the model
    void SolverFreeEnv();                                                 // free the environment

    // get obj, x, dual, rc, solver, env
    int SolverGetObjval(double *const soln);                              // get objective valur
    int SolverGetX(int start, int len, double* values);                   // get x value
    int SolverGetDual(int start, int len, double* values);                // get dual values to values
    int SolverGetRC(int start, int len, double* values);                  // get reduced cost to values

    // add variables (cols), constraints
    int SolverAddVars(int num, int numnz, int *vbeg, int *vind, double *val, double *obj, double *lb, double *ub, char *vtype, char **varname);
    // add var (col)
    int SolverAddConstr(int numz, int *const cind, double *const cval, char sense, double rhs, const char *const constrname);
    // add constraints

    GRBenv *SolverGetEnv();                                               // get env
    int SolverDelVars(int len, int *const ind);                           // delete variables, ind is the index list
    void SolverSetSolver(Solver *solver);                                 // set solver
    void SolverSetEnv(Solver *solver);                                    // set env

    GRBmodel *SolverCopy();                                               // copy model
    int SolverNewModel(const char*const Pname, int numvar, double *const obj, double *const lb, double *const ub, char *const vtype, char **varname);
                                                                          // create new model
    void printSparseMatrix(int ncols_rows, int nzcnt, int *cbeg, int *cind, double *cval, char *format);
                                                                          // print a given sparse matrix, the length of the array cbeg
// private:
    GRBmodel *model;                                                      // pointer to model
    GRBenv *env;                                                          // Gurobi environment
    double *soln;                                                         // current solution
    double objval;                                                        // current objective function value
    int solstat;                                                          // solution status: 1 => opt, 2 => unbounded, 3=> inf, 4 => inf or unbounded
    clock_t time;                                                         // the time for solving this problem
};

#endif //CODE_SOLVER_H
