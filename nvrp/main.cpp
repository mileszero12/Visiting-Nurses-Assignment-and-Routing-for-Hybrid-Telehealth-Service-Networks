#include <fstream>
#include "NurseVrp.hpp"

// #include <Python.h>
// #include <pythonrun.h>

using namespace std;

int main(){
    // Py_Initialize();

    // PyRun_SimpleString("import sys");
    // PyRun_SimpleString("sys.path.append('./RL/')");


    // PyObject* pModule = PyImport_ImportModule("RL");
//    PyObject* pFunc = PyObject_GetAttrString(pModule, "print_hello");
//    PyObject* pArgs = PyTuple_New(1);
//    PyObject* kwargs = PyDict_New();
//    PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 2));
//
//    PyObject* pReturn = PyObject_Call(pFunc, pArgs, kwargs);
//
//    int nResult;
//    PyArg_Parse(pReturn, "i", &nResult);
//    cout << "return result is " << nResult << endl;
//
//    Py_Finalize();

     NurseVrp nvrp;
     nvrp.read("10", "1");

    #ifdef BUG_PRINT_PATIENT_W
    vector<Patient> temp_p1;
    nvrp.getP1(temp_p1);
    nvrp.PatientSetScoreForWeight(temp_p1[3]); // range [0,nump1-1]

    nvrp.PatientSetScoreForWeight(temp_p1[4]); // range [0,nump1-1]

    nvrp.PatientSetScoreForWeight(temp_p1[5]); // range [0,nump1-1]

    vector<Patient> temp_p2;
    nvrp.getP2(temp_p2);
    nvrp.PatientSetScoreForWeight(temp_p2[7 - 6]); // range [nump1, nump2], but remember to minus nump1
    #endif
     nvrp.printGraph();
     nvrp.HeuristicPivotPatient();
//     nvrp.printPivot(); // print pivot patient
//     nvrp.printNurse();
     nvrp.assignPivotToNurseAndClinic();
     nvrp.assignNonPivotToNurseAndClinic();
     nvrp.ConvertLabelToRoute();
//     nvrp.LocalSearch();


    ofstream outfile("/Users/pikay/Documents/NurseRoutingProblem/vnrp/output.txt");
    int count = 0;
    for (int i = 0; i < nvrp.getRouteNum(); i++){
        for (int j = 0; j < nvrp.Seq_vlen[i]; j++){
            outfile << nvrp.Seq[count] << " ";
            count ++;
        }
        outfile << nvrp.Seq[count] << " ";
        count ++;
        outfile << nvrp.Seq[count] << " ";
        count ++;
        outfile << nvrp.Seq[count] << endl;
        count ++;
    }
    outfile.close();





    // cout << nvrp.getMaxCliqueNum() << endl;
    // nvrp.printMaxClique();
    // nvrp.printGraph();
    // nvrp.printServiceTime(1, false);

}


