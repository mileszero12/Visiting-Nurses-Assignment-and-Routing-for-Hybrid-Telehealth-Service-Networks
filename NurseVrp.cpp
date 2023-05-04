#include "NurseVrp.hpp"
#include <fstream>
#include <random>


/* ************
 * Node Class *
 ************ */

void Node::NodeSetName(int na) {
    name = na;
}

bool Node::NodeIsPatient() const {
    return Type;
}

void Node::NodeSetType(bool T) {
    Type = T;
}

void Node::NodeGetAdjPatientList(unordered_map<int, double> &vec_node) {
    vec_node = adjPatientList;
}

void Node::NodeSetLocation(double x, double y) {
    location = make_tuple(x, y);
}

double Node::travelTime(const Node &p) const {
    double xsquare = pow(NodeGetLocationx() - p.NodeGetLocationx(), 2);
    double ysquare = pow(NodeGetLocationy() - p.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELTIMERATE;
    return ceil(res * 100.0) / 100.0;
}


bool Node::NodeAddPatientToAdj(const Node &p) {
    auto before = adjPatientList.size();
    adjPatientList.insert(make_pair(p.NodeGetName(), travelTime(p)));
    auto after = adjPatientList.size();
    if (after > before) {
        return true;
    } else {
        return false;
    }
}


void Node::printAdjPatientList() {
    for (auto &p: adjPatientList) {
        cout << "Patient: " << p.first << " Travel Time: " << p.second << endl;
    }
}


bool Node::checkNodeIsInAdjList(Node *p) {
    if (adjPatientList.find(p->NodeGetName()) != adjPatientList.end()) {
        return true;
    } else {
        return false;
    }
}


double Node::getDistance(Node *p) {
    if (checkNodeIsInAdjList(p)) {
        return adjPatientList[p->NodeGetName()];
    } else {
        return 0;
    }
}


auto Node::NodeGetAdjLen() {
    return adjPatientList.size();
}

bool Node::checkAdjIsEmpty() {
    return adjPatientList.empty();
}


/* ***************
 * Patient Class *
 *************** */

double Patient::travelCost(const Node &p) {
    double xsquare = pow(NodeGetLocationx() - p.NodeGetLocationx(), 2);
    double ysquare = pow(NodeGetLocationy() - p.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOST;
    return ceil(res * 100.0) / 100.0;
}

double Patient::travelCostP(const Node &p) {
    double xsquare = pow(NodeGetLocationx() - p.NodeGetLocationx(), 2);
    double ysquare = pow(NodeGetLocationy() - p.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOSTP;
    return ceil(res * 100.0) / 100.0;
}

double Patient::travelCostP(Node *p) {
    double xsquare = pow(NodeGetLocationx() - p->NodeGetLocationx(), 2);
    double ysquare = pow(NodeGetLocationy() - p->NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOSTP;
    return ceil(res * 100.0) / 100.0;
}

void Patient::printAllTime() {
    if (!type) {
        cout << "This patient is Type 1 patient (prefer home care) " << name << endl;
        cout << "Start Time: " << get<0>(time) << endl;
        cout << "Service Time: " << get<2>(time) << endl;
        cout << "End Time: " << get<1>(time) << endl;
    } else {
        cout << "This patient is Type 1 patient (no preference) " << name << endl;
        cout << "Start Time: " << get<0>(time) << endl;
        cout << "Service Time: " << get<2>(time) << endl;
        cout << "End Time: " << get<1>(time) << endl;
    }

}

void Patient::PatientSetType(bool t) {
    type = t;
}

void Patient::PatientSetDual(double d) {
    dual = ceil(d * 100.0) / 100.0;
}

void Patient::PatientSetTime(double st, double ser, double ed) {
    time = make_tuple(ceil(st * 100.0) / 100.0, ceil(ser * 100.0) / 100.0, ceil(ed * 100.0) / 100.0);
}

void Patient::PatientInit(int na, bool kind, double lx, double ly, double st, double ser, double ed) {
    PatientSetLabelList(0);
    NodeSetName(na);
    NodeSetLocation(lx, ly);
    NodeSetType(false);
    PatientSetType(kind);
    PatientSetTime(st, ser, ed);
    cur_label_index = 0;
}

void Patient::UpdateLabelList1_Init(int cur_ind, Clinic *clinic, int t, Nurse *nurse) {
    labelList[cur_ind].setNurse(nurse);
    labelList[cur_ind].setClinic(clinic);
    labelList[cur_ind].setRouteType(t);
    labelList[cur_ind].setEndVex(NodeGetName());
}

void Patient::UpdateLabelList2_Init(int cur_ind, Node *clinic, Nurse nurse, bool isRC) {
    // update cost
    double cost = 0;

    cost += nurse.NurseGetSalary(); // nurse salary
    // cout << "Nurse Salary: " << nurse.NurseGetSalary() << endl;

    if (isRC) {
        labelList[cur_ind].setRC(PatientGetDual() + nurse.NurseGetDual());
    }

    if (type == VISITING_ROUTE) { // type 1 patient
        cost += travelCost(*clinic); // travel cost

        double st_time = travelTime(*clinic);
        if (st_time < PatientGetStartTime()) { // if travel time is smaller than start time
            st_time = PatientGetStartTime(); // the starting time is the start time
        }
        labelList[cur_ind].setST(st_time); // update starting time
    } else {
        cost += travelCostP(*clinic);

        double st_time = PatientGetStartTime(); // the starting time is the start time
        labelList[cur_ind].setST(st_time); // update starting time
    }
    // only contain one direction travelling cost and nurse salary
    labelList[cur_ind].setCost(cost);
}

void Patient::printLabelList() {
    cout << "Patient " << NodeGetName() << " has " << PatientGetLabelListLen() << " labels " << endl;
    for (int i = 0; i < PatientGetLabelListLen(); i++) {
        cout << "Label " << i << ": " << endl;
        cout << "      depot clinic: " << labelList[i].getClinic()->NodeGetName() << endl;
        if (labelList[i].getRouteType() == VISITING_ROUTE) {
            cout << "      visiting nurse: " << labelList[i].getNurse()->NurseGetName() << endl;
        } else if (labelList[i].getRouteType() == CLINIC_ROUTE) {
            cout << "      clinic nurse: " << labelList[i].getNurse()->NurseGetName() << endl;
        } else {
            cout << "      hospital nurse: " << labelList[i].getNurse()->NurseGetName() << endl;
        }
        cout << "      starting time: " << labelList[i].getST() << endl;
        cout << "      cost: " << labelList[i].getCurCost() << endl;
        cout << "      reduced cost: " << labelList[i].getRC() << endl;
        cout << "      end vertex: " << labelList[i].getEndVex() << endl;
        cout << "      parent label: " << labelList[i].getParent() << endl;
    }
}


/* **************
 * Client Class *
 ************** */

void Clinic::ClinicSetSetupFee(int fee) {
    setupfee = fee;
}

void Clinic::ClinicChangeToOpen() {
    open = true;
}

int Clinic::ClinicGetSetupFee() const {
    return setupfee;
}

bool Clinic::ClinicStatus() const {
    return open;
}

void Clinic::ClinicInit(int na, double lx, double ly, int fee) {
    NodeSetName(na);
    NodeSetType(true);
    NodeSetLocation(lx, ly);
    ClinicSetSetupFee(fee);
}

/* *************
 * Nurse Class *
 ************* */

void Nurse::NurseSetType(int t) {
    type = t;
}

void Nurse::NurseSetDual(double d) {
    dual = ceil(d * 100.0) / 100.0;
}

void Nurse::NurseSetSalary(double s) {
    salary = ceil(s * 100.0) / 100.0;
}

void Nurse::NurseSetName(int na) {
    name = na;
}

void Nurse::NurseInit(int na, int kind, double sa) {
    NurseSetName(na);
    NurseSetType(kind);
    NurseSetSalary(sa);
}

/* ****************
 * NurseVRP Class *
 **************** */
void NurseVrp::printNurse() {
    for (auto &vrp: vr) {
        cout << "Visiting nurse: " << vrp.NurseGetName() << " salary: " << vrp.NurseGetSalary() << endl;
    }
    for (auto &vrp: nr) {
        cout << "Clinic nurse: " << vrp.NurseGetName() << " salary: " << vrp.NurseGetSalary() << endl;
    }
    for (auto &vrp: hr) {
        cout << "Hospital nurse: " << vrp.NurseGetName() << " salary: " << vrp.NurseGetSalary() << endl;
    }
}

bool NurseVrp::findEle(const vector<Node> &list, const Patient &item) {
    bool flag = false;

    for (const auto &i: list) {
        if (i.NodeGetName() == item.NodeGetName()) {
            flag = true;
            break;
        }
    }

    return flag;
}

double NurseVrp::travelCost(const Node &v1, const Node &v2) {
    double xsquare = pow(v1.NodeGetLocationx() - v2.NodeGetLocationx(), 2);
    double ysquare = pow(v1.NodeGetLocationy() - v2.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOST;
    return ceil(res * 100.0) / 100.0;
}

double NurseVrp::travelCost(Node *v1, const Node &v2) {
    double xsquare = pow(v1->NodeGetLocationx() - v2.NodeGetLocationx(), 2);
    double ysquare = pow(v1->NodeGetLocationy() - v2.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOST;
    return ceil(res * 100.0) / 100.0;
}

double NurseVrp::travelDis(const Node &v1, const Node &v2) {
    double xsquare = pow(v1.NodeGetLocationx() - v2.NodeGetLocationx(), 2);
    double ysquare = pow(v1.NodeGetLocationy() - v2.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare);
    return ceil(res * 100.0) / 100.0;
}

double NurseVrp::travelTime(const Node &v1, const Node &v2) {
    double xsquare = pow(v1.NodeGetLocationx() - v2.NodeGetLocationx(), 2);
    double ysquare = pow(v1.NodeGetLocationy() - v2.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELTIMERATE;
    return ceil(res * 100.0) / 100.0;
}

double NurseVrp::travelCostPatient(const Node &v1, const Node &v2) {
    double xsquare = pow(v1.NodeGetLocationx() - v2.NodeGetLocationx(), 2);
    double ysquare = pow(v1.NodeGetLocationy() - v2.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOSTP;
    return ceil(res * 100.0) / 100.0;
}

double NurseVrp::travelCostPatient(Node *v1, const Node &v2) {
    double xsquare = pow(v1->NodeGetLocationx() - v2.NodeGetLocationx(), 2);
    double ysquare = pow(v1->NodeGetLocationy() - v2.NodeGetLocationy(), 2);
    double res = sqrt(xsquare + ysquare) * TRAVELCOSTP;
    return ceil(res * 100.0) / 100.0;
}

void NurseVrp::read(const string &instanceSize, const string &instanceNum) {
    dataSize = stoi(instanceSize);
    dataInstance = stoi(instanceNum);
    string filename = PATH_ORIGINAL + instanceSize + "/" + instanceNum + ".txt";
    fstream myfile(filename);
    #ifdef DEBUG_READ
    cout << filename << endl;
    #endif

    myfile >> numc;
    myfile >> nump1;
    myfile >> nump2;
    myfile >> numvr;
    myfile >> numnr;
    myfile >> numhr;
    nump = nump1 + nump2;

    #ifdef DEBUG_READ
    cout << "client number: " << numc << endl;
    cout << "patient number: " << nump << endl;
    cout << "type 1 patient number: " << nump1 << endl;
    cout << "type 2 patient number: " << nump2 << endl;
    #endif

    for (int i = 0; i < numc; i++) {
        Clinic tempc;
        int a, b;
        double x, y;
        myfile >> a >> x >> y >> b;
        // if this is a hospital, it needs to be opened at first
        if (i == numc - 1) {
            tempc.ClinicInit(a, x, y, b);
            tempc.ClinicChangeToOpen();
        } else {
            tempc.ClinicInit(a, x, y, b);
        }
        c.emplace_back(tempc);
    }

    numc = numc - 1;

    for (int i = 0; i < numvr; i++) {
        Nurse nvr;
        int a, type;
        double sala;
        myfile >> a >> type >> sala;
        nvr.NurseInit(a, type, sala);
        vr.emplace_back(nvr);
    }

    for (int i = 0; i < numnr; i++) {
        Nurse nnr;
        int a, type;
        double sala;
        myfile >> a >> type >> sala;
        nnr.NurseInit(a, type, sala);
        nr.emplace_back(nnr);
    }

    for (int i = 0; i < numhr; i++) {
        Nurse nhr;
        int a, type;
        double sala;
        myfile >> a >> type >> sala;
        nhr.NurseInit(a, type, sala);
        hr.emplace_back(nhr);
    }



    for (int i = 0; i < nump1; i++) {
        Patient tempp;
        int name;
        bool kind;
        double x, y, ser, st, en;
        myfile >> name >> kind >> x >> y >> ser >> st >> en;
        tempp.PatientInit(name, kind, x, y, st, ser, en);
        p1.emplace_back(tempp);
    }

    for (int i = 0; i < nump2; i++) {
        Patient tempp;
        int name;
        bool kind;
        double x, y, ser, st, en;
        myfile >> name >> kind >> x >> y >> ser >> st >> en;
        tempp.PatientInit(name, kind, x, y, st, ser, en);
        p2.emplace_back(tempp);
    }

    for (auto &p:p1){
        p.memoryAssign();
    }

    for (auto &p:p2){
        p.memoryAssign();
    }

    // from clinic, add type 1 patient and all type 2 patient to its adjacent patient list
    for (int i = 0; i < numc; i++) {
        for (int j = 0; j < nump1; j++) {
            if (travelTime(c[i], p1[j]) < p1[j].PatientGetEndTime()) {
                c[i].NodeAddPatientToAdj(p1[j]);
            }
        }
        for (int j = 0; j < nump2; j++) {
            c[i].NodeAddPatientToAdj(p2[j]);
        }
    }

    for (int j = 0; j < nump2; j++) {
        c[numc].NodeAddPatientToAdj(p2[j]);
    }

    // from type 1 patient
    for (int i = 0; i < nump1; i++) {
        // add type 1 patient
        for (int j = 0; j < nump1; j++) {
            if (p1[i].PatientGetStartTime() + p1[i].PatientGetServiceTime() + travelTime(p1[i], p1[j]) <=
                p1[j].PatientGetEndTime() && i != j) {
                p1[i].NodeAddPatientToAdj(p1[j]);
            }
        }
        // add type 2 patient
        for (int j = 0; j < nump2; j++) {
            if (p1[i].PatientGetStartTime() + p1[i].PatientGetServiceTime() + travelTime(p1[i], p2[j]) <=
                p2[j].PatientGetEndTime()) {
                p1[i].NodeAddPatientToAdj(p2[j]);
            }
        }
    }

    // from type 2 patient
    for (int i = 0; i < nump2; i++) {
        for (int j = 0; j < nump1; j++) {
            if (p2[i].PatientGetStartTime() + p2[i].PatientGetServiceTime() + travelTime(p2[i], p1[j]) <=
                p1[j].PatientGetEndTime()) {
                p2[i].NodeAddPatientToAdj(p1[j]);
            }
        }
        for (int j = 0; j < nump2; j++) {
            if (p2[i].PatientGetStartTime() + p2[i].PatientGetServiceTime() <= p2[j].PatientGetEndTime() && i != j) {
                p2[i].NodeAddPatientToAdj(p2[j]);
            }
        }
    }

}

void NurseVrp::printGraph() {
    for (int i = 0; i < numc; i++) {
        cout << "-> Clinic: " << c[i].NodeGetName() << endl;

        if (c[i].checkAdjIsEmpty()) {
            cout << "No adjacent patient" << endl;
        } else {
            c[i].printAdjPatientList();
        }
        cout << endl;
    }

    cout << "-> Hospital: " << endl;
    c[numc].printAdjPatientList();
    cout << endl;

    for (int i = 0; i < nump1; i++) {
        cout << "-> Type 1 Patient: " << p1[i].NodeGetName() << endl;

        if (p1[i].checkAdjIsEmpty()) {
            cout << "No adjacent patient" << endl;
        } else {
            p1[i].printAdjPatientList();
        }

        cout << endl;
    }

    for (int i = 0; i < nump2; i++) {
        cout << "-> Type 2 Patient: " << p2[i].NodeGetName() << endl;

        if (p2[i].checkAdjIsEmpty()) {
            cout << "No adjacent patient" << endl;
        } else {
            p2[i].printAdjPatientList();
        }
        cout << endl;
    }
}

void NurseVrp::getP1(vector<Patient> &temp_p) {
    temp_p = p1;
}

void NurseVrp::getP2(vector<Patient> &temp_p) {
    temp_p = p2;
}

void NurseVrp::getC(vector<Clinic> &temp_c) {
    temp_c = c;
}

int NurseVrp::random_select_clinic() const {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, numc - 1);
    return dis(gen);
}

void NurseVrp::initOpenClinic() {
    for (int i = 0; i < numc; i++) {
        openClinic.emplace_back(c[i].NodeGetName());
    }
}

bool NurseVrp::cmpLabelRCLess(const Label *l1, const Label *l2) {
    return l1->getRC() < l2->getRC();
}

void NurseVrp::tempCloseAClinic(vector<int> &temp_openClinic, int index) {
    for (int &it: openClinic) {
        if (it != index) {
            temp_openClinic.emplace_back(it);
        }
    }
}

void NurseVrp::closeAClinic(int index) {
    openClinic.erase(openClinic.begin() + index);
}

void NurseVrp::UpdateLabelList(Patient &p, const Patient *j, int cur_ind, int t, Label* pre_label, bool isInit) {
    // cout << "******" << pre_label->getRouteType() << endl;

    double cost = pre_label->getCurCost();
    double rc = pre_label->getRC();
    double st_time = pre_label->getST();

    // update nurse, route type, clinic, endVex
    // calculate cost
    // calculate st
    // check whether update the rc
    if (pre_label->getNurseType() == VISITING_ROUTE) {
        p.UpdateLabelList1_Init(cur_ind, &c[pre_label->getClinic()->NodeGetName()],VISITING_ROUTE,
                                &vr[pre_label->getNurse()->NurseGetName()]);
        cost += travelCost(p, *j); // cost: adding the traveling cost from patient p to patient j

        // starting time: max{pre st + pre service time + travel time form pre j to cur p, p start time}
        if (st_time + j->PatientGetServiceTime() + travelTime(p, *j) < p.PatientGetStartTime()) {
            st_time = p.PatientGetStartTime();
        } else {
            st_time += j->PatientGetServiceTime() + travelTime(p, *j);
        }

        // rc: {0(initial), pre rc + cur patient dual}
        if (!isInit) {
            rc += p.PatientGetDual();
        }
    }
    else {
        if (pre_label->getNurseType() == CLINIC_ROUTE) {
            p.UpdateLabelList1_Init(cur_ind, pre_label->getClinic(), CLINIC_ROUTE,
                                    pre_label->getNurse());
            cost += p.travelCostP(pre_label->getClinic()); // cost: adding the traveling cost from patient p to clinic

            // starting time: max{pre st + pre service time, p start time}
            if (st_time + j->PatientGetServiceTime() < p.PatientGetStartTime()) {
                st_time = p.PatientGetStartTime();
            } else {
                st_time += j->PatientGetServiceTime();
            }

            // rc: {0(initial), pre rc + cur patient dual}
            if (!isInit) {
                rc += p.PatientGetDual();
            }
        }
        else{
            p.UpdateLabelList1_Init(cur_ind, pre_label->getClinic(), HOSPITAL_ROUTE,
                                    pre_label->getNurse());
            cost += p.travelCostP(pre_label->getClinic()); // cost: adding the traveling cost from patient p to hospital

            // starting time: max{pre st + pre service time, p start time}
            if (st_time + j->PatientGetServiceTime() < p.PatientGetStartTime()) {
                st_time = p.PatientGetStartTime();
            } else {
                st_time += j->PatientGetServiceTime();
            }

            // rc: {0(initial), pre rc + cur patient dual}
            if (!isInit) {
                rc += p.PatientGetDual();
            }
        }
    }

    p.labelList[cur_ind].setCost(cost); // update cost
    p.labelList[cur_ind].setST(st_time); // update st
    p.labelList[cur_ind].setRC(rc); // update rc
    // cout << "cur_ind: " << cur_ind << endl;
    p.labelList[cur_ind].linkToOther(pre_label); // update parent
    // cout << " after linktoother: " << p.labelList[cur_ind].getParent() << endl;

    #ifdef DEBUG_CHECK_PARENT
    cout << " * In UpdateLabelList() func : Patient " << p.NodeGetName() << ", current label index is " << cur_ind;
    cout << ", its parent is " << p.labelList[cur_ind].getParent() << endl;
    // cout << endl;
    #endif

    p.PatientLabelListIndAdd(); // update label list index
    // cout << "!! here" << endl;
    // p.printLabelList();
}

int NurseVrp::getRouteNum() const{
    return n_route;
}

int NurseVrp::getCertainSeq(int i) const {
    return Seq[i];
}