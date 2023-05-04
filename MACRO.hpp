#ifndef CODE__MACRO_HPP_


//////////////////////////////////////////////////////////////////////////////
//                          Macro for NurseVrp.hpp                          //
//////////////////////////////////////////////////////////////////////////////
#define TRAVELTIMERATE 0.02
#define TRAVELCOST 2.2
#define TRAVELCOSTP 0 // 1.5
#define VISITING_ROUTE 0
#define CLINIC_ROUTE 1
#define HOSPITAL_ROUTE 2

#define SEQ_LIMIT size_t(1e12)
#define MAXROUTE 100000
#define NONZERO 1e-6
#define INIT_MEM 256
#define PATH_LARGETW "/Users/pikay/Documents/NurseRoutingProblem/NurseData/largeTW/data"
#define PATH_ORIGINAL "/Users/pikay/Documents/NurseRoutingProblem/NurseData/original/data/"
#define PATH_LOWSETUP "/Users/pikay/Documents/NurseRoutingProblem/NurseData/lowSetup/data"
const int WANCHAI_LABEL = 10000;
const int WANCHAI = 1000000000;
const int MAX = 1000; // for finding the pivot patient

#define INFEASIBLE 3
#define INFEASIBLE_OR_UNBOUNBED 4
#define FEASIBLE 2

// #define DEBUG_READ
/**
 * make sure that the data is correct after read
 */

//////////////////////////////////////////////////////////////////////////////
//                                 Heuristic                                //
//////////////////////////////////////////////////////////////////////////////
// #define DEBUG_GETMAXCLIQUE
/**
 * make sure that the number of candidate vertex is
 * correct after construct graph for max clique
 * print clique
 * ==> Print Vertex before find max clique:
 * ==> Print Vertex after find max clique:
 * ==> Check pivot patient when calculate j1_pivot_num and j2_pivot_num:
 * ==> Pivot size: 3
 * Max clique is:
 */

// #define DEBUG_GETMAXCLIQUE_ONLY
/**
 * make sure that the number of candidate vertex is
 * correct after construct graph for max clique
 * print clique
 * ==> Print Vertex before find max clique:
 * ==> Print Vertex after find max clique:
 * ==> Check pivot patient when calculate j1_pivot_num and j2_pivot_num:
 * ==> Pivot size: 3
 * Max clique is:
 */

// #define BUG_PRINT_SCORE_W1_W2
/**
 * print the score of w1 and w2 for each patient
 * do the check for the score of w1 and w2
 */

// #define DEBUG_PIVOT_CONVERT_CHECK
/**
 * print the max clique member in pivot.cpp
 */

// #define BUG_PRINT_PATIENT_W
/**
 * print the w_score of a patient, heuristic.cpp
 */

// #define DEBUG_W_SCORE
/**
 * print the w_score of all patient, heuristic.cpp
 */

// #define DEBUG_PRINT_PIVOT_AFTER_2CRITERION
/**
 * print the pivot patient after 2 criterion, heuristic.cpp
 */

// #define DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
/**
 * print w_score
 * print backupPatient
 * print whether get in the second loop for making the number of pivot to be enough
 * print the j1_pivot_num and j2_pivot_num
 * print pivotPatient and pivotPatientSet
 */

// #define DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
// #define DEBUG_PRINT_PIVOT_BY_PATIENT_TYPE
/**
 * print the pivot patient by patient type, heuristic.cpp
 * Type 1: ...
 * Type 2: ...
 */

//////////////////////////////////////////////////////////////////////////////
//                              MaxClique                                   //
//////////////////////////////////////////////////////////////////////////////
// #define DEBUG_FIND_EDGE
/**
 make sure that the edges are correct after
 * construct graph for max clique
 * print the pair of edges
 * example: J1 to J1: 2, 3
 */


// #define DEBUG_CLIQUES
/**
 * make sure that the vertex is correct after
 * construct graph for max clique
 * print the procedure in findCliques() function
 * and select_random() function
 */

// #define MAX_CLIQUE_PIVOT_CHECK
/**
 * print criteria 1 for pivot patient
 * E_i + T_ij > L_j
 * E_j + T_ji > L_i
 */

// #define DEBUG_CHECK_VECTOR_LIST
/**
 * check the VertexList is empty or not
 */


//////////////////////////////////////////////////////////////////////////////
//                            Initialization                                //
//////////////////////////////////////////////////////////////////////////////
// #define DEBUG_PRINT_PIVOT_LABEL_LIST
/**
 * print the "pivot" patient's label list
 * in assignPivotToNurseAndClinic() function
 * print label list contain: depot clinic, nurse, starting time, cost, reduced cost, end vertex
 */

// #define DEBUG_PRINT_NON_PIVOT_LABEL_LIST

// #define DEBUG_ADD_LABEL
/**
 * print the label list in addLabel() function
 */

// #define DEBUG_CHECK_PARENT
/**
 * print the parent of a label in curretnP_AddLabel() function
 */

// #define DEBUG_PRINT_NON_PIVOT_PATIENT
/*
 * Example:
 * Type 1 Patient: 0 is not pivot patient
 * Type 1 Patient: 2 is not pivot patient
 */


// #define DEBUG_PRINT_NON_PIVOT_LABEL_LIST2


// #define CODE__MACRO_HPP_

// #define DEBUG_UPDATE_ROUTE

#define CHECK_ROUTE_IN_LABEL_TO_ROUTE
/**
 * check the all route in ConvertlabelToRoute() function
 * Route 5 has 2 element.
 * Patient: 5 2
 * clinic: 2
 * route type: 0
 * nurse: 3

 */

// #define DEBUG_UPDATE_ROUTE_NURSE_CHECK_VIND
/*
 * visiting tempNurse Index: 2
   clinic tempNurse Index: 0
   visiting tempNurse Index: 1 ...
 */

#endif //CODE__MACRO_HPP_
