#include "Solver.hpp"

using namespace std;


int Solver::SolverGetRowNum(int *const pvalue) {
    return GRBgetintattr(model, "NumConstrs", pvalue);
}

int Solver::SolverGetColNum(int *const pvalue){
    return GRBgetintattr(model, "NumVars", pvalue);
}

int Solver::SolverGetStatus(int *const pvalue){
    return GRBgetintattr(model, "Status", pvalue);
}

int Solver::SolverOptimize(){
    return GRBoptimize(model);
}

int Solver::SolverUpdate(){
    return GRBupdatemodel(model);
}

void Solver::SolverFreeModel(){
    GRBfreemodel(model);
}

void Solver::SolverFreeEnv(){
    GRBfreeenv(env);
}

int Solver::SolverGetObjval(double *const soln){
    return GRBgetdblattr(model, "ObjVal", soln);
}

int Solver::SolverGetX(int start, int len, double *values){
    return GRBgetdblattrarray(model, "X", start, len, values);
}

int Solver::SolverGetDual(int start, int len, double *values) {
    return GRBgetdblattrarray(model, "Pi", start, len, values);
}

int Solver::SolverGetRC(int start, int len, double *values){
    return GRBgetdblattrarray(model, "RC", start, len, values);
}

int Solver::SolverAddVars(int num, int numnz, int *vbeg, int *vind, double *val, double *obj, double *lb, double *ub, char *vtype, char **varname){
    return GRBaddvars(model, num, numnz, vbeg, vind, val, obj, lb, ub, vtype, varname);
}

int Solver::SolverAddConstr(int numz, int *const cind, double *const cval, char sense, double rhs, const char *const constrname){
    return GRBaddconstr(model, numz, cind, cval, sense, rhs, constrname);
}

GRBenv *Solver::SolverGetEnv(){
    return env;
}

int Solver::SolverDelVars(int len, int *const ind){
    return GRBdelvars(model, len, ind);
}

void Solver::SolverSetSolver(Solver *solver){
    model = solver->SolverCopy();
    env = solver->SolverGetEnv();
}

void Solver::SolverSetEnv(Solver *solver){
    env = solver->SolverGetEnv();
}

GRBmodel *Solver::SolverCopy(){
    return GRBcopymodel(model);
}

int Solver::SolverNewModel(const char *const Pname, int numvar, double *const obj, double *const lb, double *const ub, char *const vtype, char **varname){
    return GRBnewmodel(env, &model, Pname, numvar, obj, lb, ub, vtype, varname);
}

void Solver::printSparseMatrix(int ncols_rows, int nzcnt, int *cbeg, int *cind, double *cval, char *format){
    cout << endl << "Print Sparse Matrix:" << endl;
    cout << "Number of" << format << "=" << ncols_rows << endl;

    for (int i = 0; i < ncols_rows - 1; i++) {
        cout << format << ": " << endl;
        for (int j = cbeg[i]; j < cbeg[i+1]; j++) {
            cout << "Index = " << cind[j] << "Value = " << cval[j] << endl;
        }
        cout << endl;
    }


    for (int j = cbeg[ncols_rows - 1]; j < nzcnt; j++) {
        cout << "Index = " << cind[j] << "Value = " << cval[j] << endl;
    }
    cout << endl << endl;

}










