#ifndef NurseVrp_hpp
#define NurseVrp_hpp

#include <tuple>
#include <vector>
#include <cstdio>
#include <string>
#include <iostream>
#include <cmath>
#include "gurobi_c.h"
#include "gurobi_c++.h"
#include "Solver.hpp"
#include <set>
#include <unordered_map>
#include "MACRO.hpp"
#include <queue>
#include <map>


using namespace std;

/* Class for vetex */
class Node {
public:
    void NodeSetName(int na);

    /**
     * @brief Set the Node Name object
     */

    bool NodeIsPatient() const;

    /**
     * @brief check whether this node is a patient or not,
     * return true is this node is a patient, else return false
     */

    void NodeSetType(bool T);

    /**
     * @brief Set the type of this node as patient/clinic
     * @param T the type of the node, true for patient, false for clinic
     */

    auto NodeGetType() const { return Type; }

    /**
     * @brief Get the type of this node
     * @return the type of this node, true for patient, false for clinic
     */

    auto NodeGetName() const { return name; }

    /**
     * @brief Get the Name of the node
     * @return the name of the node
     */

    void NodeSetLocation(double x, double y);

    /**
     * @brief Set the location of the node
     * @param x the x coordinate of the node
     * @param y the y coordinate of the node
     */

    auto NodeGetLocationx() const { return get<0>(location); }

    /**
     * @brief Get the x coordinate of the node
     * @return the x coordinate of the node
     */

    auto NodeGetLocationy() const { return get<1>(location); }

    /**
     * @brief Get the y coordinate of the node
     * @return the y coordinate of the node
     */

    void NodeGetAdjPatientList(unordered_map<int, double> &vec_node);

    /*
     * @brief Get the adjacent patient list of this node
     * @param adjPatientList the adjacent patient list of this node, pass the reference of the list
     */

    void printAdjPatientList();

    /**
     * @brief Print the adjacent patient list of this node
     */

    bool NodeAddPatientToAdj(const Node& p);

    /**
     * @brief Add a patient to the adjacent patient list of this node
     * @param p the Patient to be added, data type is Node
     * @return true if the patient is added successfully, else return false
     */

    auto NodeGetAdjLen();

    /**
     * @brief Get the length of the adjacent patient list of this node
     * @return the length of the adjacent patient list of this node
     */

    double travelTime(const Node& p) const;

    /**
     * @brief Calculate the travel time between this node and the patient p
     * @param p the Patient (Clinic) to be calculated, data type is Node
     * @return the travel time between this node and the patient p
     */

    double getDistance(Node *p);

    /**
     * @brief Get the travel time between this node and the patient p
     * @param p the Patient (Clinic) to be calculated, data type is Node
     * @return the travel time between this node and the Patient p, if the patient is not adjacent to this node, return 0
     */

    bool checkNodeIsInAdjList(Node *p);

    /**
     * @brief Check whether the Patient p is in the adjacent patient list of this node
     * @param p the Patient to be checked, data type is Node
     * @return true if the patient p is in the adjacent patient list of this node, else return false
     */

    bool checkAdjIsEmpty();
    /**
     * @brief Check whether the adjacent patient list of this node is empty
     * @return true if the adjacent patient list of this node is empty, else return false
     */

protected:
    int name{};                                              // the name of the node
    bool Type{};                                             // the type of the node, true for patient, false for clinic
    tuple<int, int> location;                                // the location of the node <x, y>
    unordered_map<int, double> adjPatientList;
    /**
     * @brief the adjacent patient list of this node, the key is the name of the patient,
     * the value is the travel time between this node and the patient, <patient index, travel time>
     */
};

/* Class for nurse */
class Nurse {
public:
    void NurseSetDual(double d);

    /**
     * @brief Set the dual of the nurse
     * @param d the dual of the nurse
     */

    void NurseSetSalary(double s);

    /**
     * @brief Set the salary of the nurse
     * @param s the salary of the nurse
     */

    void NurseSetType(int t);

    /**
     * @brief Set the type of the nurse
     * @param t the type of the nurse, 0 for vr, 1 for nr, 2 for hr
     */

    void NurseSetName(int na);

    /**
     * @brief Set the name of the nurse
     * @param na the name of the nurse
     */

    [[nodiscard]] auto NurseGetDual() const { return dual; }

    /**
     * @brief Get the dual of the nurse
     * @return the dual of the nurse
     */

    [[nodiscard]] auto NurseGetSalary() const { return salary; }

    /**
     * @brief Get the salary of the nurse
     * @return the salary of the nurse
     */

    [[nodiscard]] auto NurseGetName() const { return name; };

    /**
     * @brief Get the name of the nurse
     * @return the name of the nurse
     */

    [[nodiscard]] auto NurseGetType() const { return type; }

    /**
     * @brief Get the type of the nurse
     * @return the type of the nurse, 0 for vr, 1 for nr, 2 for hr
     */

    void NurseInit(int name, int type, double sa);
    /**
     * @brief Initialize the nurse
     * @param name the name of the nurse
     * @param type the type of the nurse, 0 for vr, 1 for nr, 2 for hr
     * @param sa the salary of the nurse
     */

     Nurse(){name = 0, type = 0, salary = 0, dual = -1;}

private:
    int name;                                                // the name of the nurse
    int type;                                                // the type of the nurse, 0 for vr, 1 for nr, 2 for hr
    double salary;                                           // the salary of the nurse
    double dual;                                             // dual of nurse

};

/* Class for clinic, derived from node class
   Notice that clinic[numc] is hospital whose setup fee is 0. */
class Clinic : public Node {
public:
    void ClinicSetSetupFee(int fee);

    /**
     * @brief Set the setup fee of the clinic
     * @param fee the setup fee of the clinic
     */

    void ClinicChangeToOpen();

    /**
     * @brief Change the clinic to open
     */

    int ClinicGetSetupFee() const;

    /**
     * @brief Get the setup fee of the clinic
     * @return the setup fee of the clinic
     */

    bool ClinicStatus() const;

    /**
     * @brief Get the status of the clinic
     * @return the status of the clinic, true for open, false for close
     */

    void ClinicInit(int na, double lx, double ly, int fee);
    /**
     * @brief Initialize the clinic
     * @param na the name of the clinic
     * @param lx the x coordinate of the clinic
     * @param ly the y coordinate of the clinic
     * @param fee the setup fee of the clinic
     */

private:
    int setupfee{};                                           // the setup fee of the clinic
    bool open = false;                                      // the status of the clinic, true for open, false for close
};

/* Class for labeling */
class Label {
public:
    void linkToOther(Label *pre);
    /**
     * @brief link the current label to the previous label
     * @param pre the previous label
     */

    void setRouteType(int t) { route_type = t; }
    /**
     * @brief set the route type of the current label
     * @param t the route type
     */

    Label *getParentLabel();
    /**
     * @brief get the parent label
     * @return the previous label
     */

    void setRC(double new_rc) { cur_rc = new_rc; }

    /**
     * @brief add the reduced cost to the current label
     * @param new_rc the new reduced cost
     */

    void setCost(double new_cost) { cur_cost = new_cost; }

    /**
     * @brief add the cost to the current label
     * @param new_cost the cost to be added
     */

    void setST(double new_st) { cur_st = new_st; }

    /**
     * @brief set the start time of the current label
     * @param new_st the new start time
     */

    void setClinic(Clinic* new_clinic) { st_depot = new_clinic; }

    /**
     * @brief set the clinic to be the start depot
     * @param new_clinic the clinic number
     */

    void setNurse(Nurse* new_nurse) {cur_nurse = new_nurse;}
    /**
     * @brief set the nurse number
     * @param new_nurse the nurse number
     */

    // void setIsRoute() {is_route = true;}
    /**
     * @brief set the label as a route
     * set the is_route to be true
     */

    // void setIsntRoute() {is_route = false;}
    /**
     * @brief set the label as not a route
     * set the is_route to be false
     */

    // Const member functions
    [[nodiscard]] auto getRouteType() const { return route_type; }

    /**
     * @brief get the route type of the label
     * @return the route type of the label, int
     */

    [[nodiscard]] auto getCurCost() const { return cur_cost; }

    /**
     * @brief get the current cost of the label
     * @return the current cost of the label, double
     */

    [[nodiscard]] auto getRC() const { return cur_rc; }

    /**
     * @brief get the reduced cost of the label
     * @return the reduced cost of the label, double
     */

    [[nodiscard]] auto getST() const { return cur_st; }

    /**
     * @brief get the starting time of the label
     * @return the starting time of the label, double
     */

    [[nodiscard]] auto getNurse() const { return cur_nurse; }
    int checkNurse() { if (cur_nurse == nullptr) {return -1;} else {return cur_nurse->NurseGetName();}}
    /**
     * @brief get the nurse assigned to this label
     * @return the nurse, int
     */

    [[nodiscard]] auto getNurseType() const { return cur_nurse->NurseGetType(); }

    /**
     * @brief get the type of the nurse assigned to this label
     * @return the type of the nurse, int
     */

    [[nodiscard]] auto getClinic() const { return st_depot; }
    int checkClinic() { if (st_depot == nullptr) {return -1;} else {return st_depot->NodeGetName();}}
    /**
     * @brief get the clinic of the label
     * @return the clinic, int
     */

    void setEndVex(int end) { EndVex = end; }

    /**
     * @brief set the end vertex of the label
     * @param end the end vertex
     */

    [[nodiscard]] auto getEndVex() const { return EndVex; }

    /**
     * @brief get the end vertex of the label
     * @return the end vertex
     */

    int getParent();

    /**
     * @brief get the parent of the label
     * @return the parent of the label
     * since the end vertex is the current vetex, we can just print the end vertex
     */

    Label() {
        route_type = -1;
        cur_rc = 0;
        cur_cost = 0;
        cur_st = -1;
        st_depot = nullptr;
        cur_nurse = nullptr;
        // is_route = true;
        EndVex = -1;
        parent = nullptr;
    }
    /**
     * @brief constructor of Label
     * assume the patient of this label is the first patient of the route
     * this is a route
     */

    // ~Label();

private:
    int route_type;          // 0: visiting route, 1: clinic route, 2: hospital route
    double cur_rc;           // current reduced cost
    double cur_cost;         // current cost
    double cur_st;           // current start time
    Clinic *st_depot;        // start depot
    Nurse *cur_nurse;        // current nurse num
    // bool is_route;        // whether the label is a route
    // start from 0 and indicates the index of the last element of sequence
    int EndVex;              // end vertex of the route

    Label *parent{};         // parent label
};

/* Class for patient, derived from node class */
class Patient : public Node {
public:
    double travelCost(const Node& p);

    /**
     * @brief Calculate the travel time between this patient and the node p
     * @param p clinic or patient, data type is Node
     * @return the nurse travel cost between this patient and the clinic/patient p
     */

    double travelCostP(const Node& p);
    double travelCostP(Node *p);
    /**
     * @brief Calculate the travel cost between this patient and node p
     * @return the patient travel cost between this patient and the node p
     */

    auto PatientGetStartTime() const { return get<0>(time); }

    /**
     * @brief Get the start time of the patient
     * @return the start time of the patient
     */

    auto PatientGetEndTime() const { return get<2>(time); }

    /**
     * @brief Get the end time of the patient
     * @return the end time of the patient
     */

    auto PatientGetServiceTime() const { return get<1>(time); }

    /**
     * @brief Get the service time of the patient
     * @return the service time of the patient
     */

    void printAllTime();

    /**
     * @brief Print the start time, end time and service time of the patient
     */


    auto PatientGetType() const { return Type; }

    /**
     * @brief Get the type of the patient
     * @return the type of the patient, true for type 1 (preference), false for type 2(no preference)
     */

    void PatientSetType(bool t);

    /**
     * @brief Set the type of the patient
     * @param t the type of the patient, true for type 1 (preference), false for type 2(no preference)
     */

    void PatientSetDual(double d);

    /**
     * @brief Set the dual of the patient
     * @param d the dual of the patient
     */

    auto PatientGetDual() const { return dual; }

    /**
     * @brief Get the dual of the patient
     * @return the dual of the patient
     */

    void PatientSetTime(double st, double ser, double ed);

    /**
     * @brief Set the start time, service time and end time of the patient
     * @param st the start time of the patient
     * @param ser the service time of the patient
     * @param ed the end time of the patient
     */

    void PatientSetLabelList(int ind) { cur_label_index = ind; }

    /**
     * @brief Set the current label index of the label list for this patient
     * @param ind the current label index of the patient
     */

    void PatientLabelListIndAdd() { cur_label_index += 1; }

    /**
     * @brief Add 1 to the current label index of the label list for this patient
     */

    int PatientGetLabelListLen() const { return cur_label_index; }

    /**
     * @brief Get the current label index of the label list for this patient
     * @return the current label index of the label list for this patient
     */

    void PatientInit(int na, bool kind, double lx, double ly, double st, double ser, double ed);

    /**
     * @brief Initialize the patient
     * @param na the name of the patient
     * @param kind the type of the patient, true for type 1 (preference), false for type 2(no preference)
     * @param lx the x coordinate of the patient
     * @param ly the y coordinate of the patient
     * @param st the start time of the patient
     * @param ser the service time of the patient
     * @param ed the end time of the patient
     */

    void UpdateLabelList1_Init(int cur_ind, Clinic *clinic, int t, Nurse *nurse);

    /**
     * @brief Update a certain label list of the patient, only update type, clinic, nurse, endVex
     * @param t the type of the label
     * @param clinic the clinic
     * @param nurse the nurse who serve the patient
     */

    void UpdateLabelList2_Init(int cur_ind, Node *clinic, Nurse nurse, bool isRC);

    /**
     * @brief Update a certain label list of the patient, only update cost, start time
     * @param cur_ind the current label index of the patient
     * @param t the type of the label
     * @param clinic the clinic
     * @param nurse the nurse who serve the patient
     * @param isRC true for update reduced cost (cur patient + nurse)
     * @note the cost only contain "one" direction travelling cost and "nurse salary"
     */


    void printLabelList();
    /**
     * @brief Print the label list of the patient
     */


    void memoryAssign() {labelList= new Label[WANCHAI_LABEL];}
    /**
     * @brief Assign memory for the label list of the patient
     * call this function after read p data, use for loop
     */

    void releaseMemory() const {delete [] labelList;}
    /**
     * @brief Release the memory of the label list of the patient
     * call this function at the end, use for loop
     */

public:
    bool type{};                                          // the type of the patient
                                                          // true for type 1 (preference)
                                                          // false for type 2(no preference)

    tuple<double, double, double> time;                   // the start time, service time and end time of the patient
                                                          // <start time, service time, end time>

    double dual{};                                        // the dual of the patient
    int cur_label_index{};                                // the current label index of the patient
    Label *labelList;
};


struct GurobiSolver { // use C++ to construct model
    double *soln;                                                         // current solution
    double objval;                                                        // current objective function value
    int solstat;                                                          // solution status: 1 => opt, 2 => unbounded, 3=> inf, 4 => inf or unbounded
    clock_t time;                                                         // the time for solving this problem
};

/* score for initialization criteria 2 weight */
struct Score {
    int patient{};
    double weight{};

    Score(int p, double w) { patient = p, weight = w; }
};

/* compare function for sorting score in criteria 2 */
struct CompareScore {
    bool operator()(Score *a, Score *b) {
        return a->weight < b->weight;
    }
};

/* distance to hospital for initialization finding feasible route */
struct DisToHos {
    int patient;
    double dis;

    DisToHos(int i, double d) { patient = i, dis = d; }
};

/* compare function for sorting distance to hospital in initialization */
struct CompareDis {
    bool operator()(DisToHos const &p1, DisToHos const &p2) {
        return p1.dis > p2.dis;
    }
};

/* Class for whole problem, NurseVrp */
class NurseVrp {
public:
    NurseVrp() {
        graph = new int*[MAX];
        // Allocate memory for each column in each row
        for (int i = 0; i < MAX; i++) {
            graph[i] = new int[MAX];
        }
        Seq = new int[SEQ_LIMIT];
        Seq_vlen = new int[MAXROUTE];
        Seq_obj = new double[MAXROUTE];
        Seq_vbeg = new int[MAXROUTE];
        Seq_vval = new int[SEQ_LIMIT];
        Seq_vind = new int[SEQ_LIMIT];
    }
    void getP1(vector<Patient> &temp_p);

    /**
     * @brief Get the Type 1 patients in the problem
     * @param temp_p pass a reference of a vector of patients to store the type 1 patients in the problem
     */

    void getP2(vector<Patient> &temp_p);

    /**
     * @brief Get the Type 2 patients in the problem
     * @param temp_p pass a reference of a vector of patients to store the type 2 patients in the problem
     */

    void getC(vector<Clinic> &temp_c);

    /**
     * @brief Get the clinics in the problem
     * @param temp_c pass a reference of a vector of clinic to store the clinic in the problem
     */

    void printNurse();
    /**
     * @brief Print the nurses in the problem
     */

/////////////////////////////////////////////////////////////////////
//   function to calculate the distance or cost between two node   //
/////////////////////////////////////////////////////////////////////
    static double travelDis(const Node& v1, const Node& v2);

    /**
     * @brief Calculate the distance between two node
     * @param v1 the first node
     * @param v2 the second node
     * @return the distance between two node
     */

    static double travelCost(const Node& v1, const Node& v2);
    static double travelCost(Node *v1, const Node& v2);
    /**
     * @brief Calculate the travel cost between two node
     * @param v1 the first node
     * @param v2 the second node
     * @return the cost between two node
     */

    static double travelCostPatient(const Node& v1, const Node& v2);
    static double travelCostPatient(Node *v1, const Node& v2);
    /**
     * @brief Calculate the travel cost between two node for patien
     * @param v1 the first node
     * @param v2 the second node
     * @return the cost between two node
     */

    static double travelTime(const Node& v1, const Node& v2);

    /**
     * @brief Calculate the travel time between two node
     * @param v1 the first node
     * @param v2 the second node
     * @return the time between two node
     */


    static bool findEle(const vector<Node>& list, const Patient& item);

    /**
     * @brief Find the item (Node) in the list
     * @param list the list to be searched
     * @param item the item to be found
     * @return true if the item is in the list, false if not
     */

    void read(const string& instanceSize, const string& instanceNum);

    /**
     * @brief Read the instance from the file
     * @param instanceSize the size of the instance
     * @param instanceNum the number of the instance
     */

    void printGraph();

    /**
     * @brief Print the graph for original problem
     */

    [[nodiscard]] int random_select_clinic() const;

    /**
     * @brief Randomly select a clinic
     * @return the index of the clinic
     */

    void initOpenClinic();

    /**
     * @brief Set the open clinic
     */

    void closeAClinic(int index);

    /**
     * @brief Close a clinic
     * @param index the index of the clinic
     */

    void tempCloseAClinic(vector<int> &temp_openClinic, int index);
    /**
     * @brief Temporarily close a clinic
     * @param temp_openClinic the vector of the temporary open clinic (output)
     * @param index the index of the clinic
     * use this function to find the optimal elimination clinic
     * then call closeAClinic() to close the optimal clinic (update real open clinic)
     */

//////////////////////////////////////////////////////////////////////
///                           Max Clique                            //
//   construct the graph first, then call findCliques() function    //
//////////////////////////////////////////////////////////////////////
    void ConstructEdgeForMaxClique();

    /**
     * @brief Construct the edge for max clique
     */

    auto select_random();

    /**
     * @brief Select a random number in the vertex set for max clique
     * @return a random vertex
     */

    void findCliques();

    /**
     * @brief Find the maximum cliques in the graph
     */

    int getMaxCliqueNum();

    /**
     * @brief Get the maximum clique number
     * @return the number of vertex of maximum clique
     */

    void printEdge();

    /**
     * @brief Print the edge of the graph for max clique
     */

    void printVertex();

    /**
     * @brief Print the vertex of the graph for max clique
     */

    void printMaxClique();

    /**
     * @brief Print the vertex of the maximum clique
     */

    void cliqueToPivot();
    /**
     * @brief Convert the maximum clique to pivot patient type
     * clique set is the vector<int>
     * pivot patient type is the vector<Patient>
     */

//////////////////////////////////////////////////////////////////////////
//                          Find Pivot Patient                          //
//////////////////////////////////////////////////////////////////////////
    void printPivot();

    /**
     * @brief Print the pivot patient
     */

    void PatientSetScoreForWeight(Patient p);

    /**
     * @brief Set the w score for patient
     * Note: the size of patient is fixed,
     * but the size of clinic is depend on the open clinic
     */

    void printWScore();

    /**
     * @brief Print the w score for patient
     */

    void HeuristicFindPivotCriterion1(set<int> &pivotPatientSet);

    /**
     * @brief Heuristic find pivot patient
     * Criterion 1: the impatible patient, max clique
     * the pivotPatient won't be updated
     */

    void HeuristicFindPivotCriterion2(set<int> &pivotPatientSet);

    /**
     * @brief Heuristic find pivot patient
     * Criterion 2: the patient with the most score
     * the w_score will be updated, and then pop some of it
     * the pivotPatient will be updated
     */

    static void print_queue_patient(queue<Patient> q);

    /**
     * @brief Print the patient in the queue, for backupPatient queue
     */

    void HeuristicUpdateNonPivotPatient(set<int> &pivotPatientSet);

    /**
     * @brief Heuristic update the non pivot patient
     * the nonPivotPatient will be updated
     */

    void HeuristicPivotPatient();
    /**
     * @brief Find the pivot patient by heuristic algorithm
     */


//////////////////////////////////////////////////////////////////////////////
//                          Find Init                                       //
//////////////////////////////////////////////////////////////////////////////
    void assignPivotToNurseAndClinic();

    /**
     * @brief Assign the pivot patient to nurse and clinic
     */

    void assignNonPivotToNurseAndClinic();

    /**
     * @brief Assign the non pivot patient to current route
     */

    void currentP_AddLabel(Patient &p);

    /**
     * @brief Insert the current patient to the current route
     * adding labels to the current patient
     * @param p the patient to be inserted
     */

    static bool cmpLabelRCLess(const Label *l1, const Label *l2);
    /**
     * @brief Compare the label by the reduce cost
     * @param l1 the first label
     * @param l2 the second label
     * @return true if the reduce cost of l1 is less than l2, false if not
     */
///////////////////////////////////////////////////////////////////////////////
//                          Find Init                                       //
//////////////////////////////////////////////////////////////////////////////
    void UpdateLabelList(Patient &p, const Patient* j, int cur_ind, int t, Label* pre_label, bool isInit);

    /**
     * @brief Update a certain label list of the patient, update type, clinic, nurse, endVex, cost, reduce cost, start time
     * @param p the patient to be inserted
     * @param j the patient, j->p, p's parent point to j's label
     * @param t the type of the label
     * @param label the label to be continued, j's label
     * @param flag the flag to indicate whether the current is initialization (True), else false
     */


    void ConvertLabelToRoute();
    /**
     * @brief Convert the label to route
     */

    void updateRoute(int &SeqIndex, int &vval_index, int &SeqIndexBegin, Patient *p);

    // void GurobiSolve(GurobiSolver gs);                // solve this vrp problem by gurobi
    // void HeuristicSolve(Solver HeutisticSolve);       // solve this vrp problem by heuristic algorithm
    // void BCPSolve();                                  // solve this vrp problem by branch-cut-and-price
    // void BilevelSolve();                              // solve this vrp problem by bi-level algorithm (local search)

    // print the key parameter in initiliazation
    void printRoute();

    void printVLen();

    void printVBeg();

    void printObj();

    void LocalSearch();

    void Insertion();

    void Deletion();

    void Relocation();



    ~NurseVrp() {

        for (int i = 0; i < nump1; ++i) {
            p1[i].releaseMemory();
        }
        for (int i = 0; i < nump2; ++i) {
            p2[i].releaseMemory();
        }

        delete[] Seq;
        delete[] Seq_vlen;
        delete[] Seq_vval;
        delete[] Seq_vbeg;
        delete[] Seq_vind;
        delete[] Seq_obj;
    }


    void printCertainRoute(int rIndex);

    int getRouteNum() const; // get the number of route

    int getCertainSeq(int i) const; // get the certain element of the seq


    int dataSize{};                                        // store the size of the data
    int dataInstance{};                                    // store the index of the data
    int numc{};                                            // store the number of clinic, not include hospital
    int nump1{};                                           // store the number of the Type 1 patient
    int nump2{};                                           // store the number of the Type 2 patient
    int nump{};                                            // store the total number of patient
    int numvr{};                                           // store the number of visiting nurse
    int numnr{};                                           // store the number of clinic nurse
    int numhr{};                                           // store the number of hospital nurse

    // lists of patient
    vector<Patient> p1;                                    // a list of Type 1 patient
    vector<Patient> p2;                                    // a list of Type 2 patient

    // lists of nurse
    vector<Nurse> vr;                                       // a list of visiting nurse
    vector<Nurse> nr;                                       // a list of clinic nurse
    vector<Nurse> hr;                                       // a list of hospital nurse

    // lists of clinic and hospital
    vector<Clinic> c;                                        // a list of clinics and hospital, the last one is the hospital
    vector<int> openClinic;                                  // a list of open clinics index

    // store the information of the MaxClique
    // int graph[MAX][MAX];                                  // graph for max clique (prepare)
    int **graph;                                             // graph for max clique (all candidate)
    set<int> vertexList;                                     // store the vertex of the above graph (all candidate)
    vector<tuple<int, int>> edges;                           // store the edges of the above graph
    int edgeNum{};                                           // store the number of edges
    vector<int> clique;                                      // store the maximum clique vertex **

    // store the information of the pivot
    vector<int> pivotPatient;                                //store the index of the pivot patient
    vector<int> nonPivotPatient;                             //store the index of the non-pivot patient
    priority_queue<Score *, vector<Score *>, CompareScore> w_score;
    /**
     * @brief store the weight score of the pivot patient, <patient index, weight score>
     * @param key the index of the pivot patient
     * @param value the weight score of the pivot patient
     */


    int *Seq{};                                               // store the sequence of the route

    int n_route{};                                            // store the number of route

    int *Seq_vlen{};                                          // store the size of each route
    int *Seq_vind{};                                          // store the non-zero constraint index of each route
    int *Seq_vval{};                                          // store the non-zero value of each route 1....
    int *Seq_vbeg{};                                          // store the first non-zero value index of each route
    double *Seq_obj{};                                        // store the cost of each route

    int LocalSearchNum = 10;                                       // store the maximal number of local search iteration
};

#endif /* NurseVrp_hpp */
