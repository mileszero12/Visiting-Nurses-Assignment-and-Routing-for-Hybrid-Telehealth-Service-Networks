#include "NurseVrp.hpp"
#include <queue>

///////////////////////////////////////////////////
//           find pivot patient                  //
///////////////////////////////////////////////////
void NurseVrp::PatientSetScoreForWeight(Patient p) {
    double w1 = 0;
    int count = 0;
    double w2 = 12;

    set<int> visitedAdjList;  // store the visited adjacent node

    // from p to others
    unordered_map<int, double> adjNodeListForP;  // create a map to store the adjacent node and its distance to p
    p.NodeGetAdjPatientList(adjNodeListForP);  // pass it to the function to get the adjacent node list of p
    for (auto & it : adjNodeListForP) {
        visitedAdjList.insert(it.first);  // insert the adjacent node to the visited list
        w1 += it.second;  // add the distance to w1
        if (w2 > it.second) { // w2 is the minimal distance
            w2 = it.second;  // update w2
        }
        count += 1;
    }


    // from others to p, they are not sysmetric
    for (auto &p_2 : p2) {
        // p_2 != p, p is in the adjList of p_2, and p_2 is not visited
        if (p_2.NodeGetName() != p.NodeGetName()) {
            if (p_2.checkNodeIsInAdjList(&p) && visitedAdjList.find(p_2.NodeGetName()) == visitedAdjList.end()) {
                visitedAdjList.insert(p_2.NodeGetName());   // insert the adjacent node to the visited list
                w1 += p_2.getDistance(&p);
                if (p_2.getDistance(&p) < w2) {   // if p_2 => p distance < w2
                    w2 = p_2.getDistance(&p);    // update w2
                }
                count += 1;
            }
        }
    }

    for (auto &p_2 : p1) {
        // p is in the adjList of p_2, and p_2 is not visited
        if (p_2.NodeGetName() != p.NodeGetName()) {
            if (p_2.checkNodeIsInAdjList(&p) && visitedAdjList.find(p_2.NodeGetName()) == visitedAdjList.end()) {
                visitedAdjList.insert(p_2.NodeGetName());   // insert the adjacent node to the visited list
                w1 += p_2.getDistance(&p);
                if (p_2.getDistance(&p) < w2) {
                    w2 = p_2.getDistance(&p);
                }
                count += 1;// only add one to the weight
            }
        }
    }

    set<int> visitedAdjList2;  // store the visited adjacent node(clinic)
    w1 = w1 / count;
    double w = w1 * w2;
    // cout << "w1: " << w1 << " w2: " << w2 << " w: " << w << endl;

    #ifdef BUG_PRINT_SCORE_W1_W2
    cout << "Patient " << p.NodeGetName() << " w1_score: " << w1 << " count: " << count << " w2_score:" << w2 << " w:" << w << endl;
    #endif

    auto *s = new Score(p.NodeGetName(), w);
    // cout << "==" << w << endl;

    w_score.emplace(s);


#ifdef BUG_PRINT_PATIENT_W
    cout << "Adjacent node list of patient " << p.NodeGetName() << " is: " << endl;
    for (auto &adjNode : visitedAdjList2) {
        if (adjNode == numc) {
            cout << "Hospital " << "  ";
        } else {
            cout << "Clinic " << adjNode << "  ";
        }
    }
    for (auto &adjNode : visitedAdjList) {
        cout << "Patient " << adjNode << "  ";
    }
    cout << endl;

    cout << "Patient " << p.NodeGetName() << " w1_score: " << w1 << endl;
    cout << "Patient " << p.NodeGetName() << " w2_score: " << w2 << endl;
    cout << "Patient " << p.NodeGetName() << " w_score: " << w << endl;
    cout << endl;
#endif
}

void NurseVrp::printPivot() {
    cout << "Pivot Patient: " << endl;
    for (int i : pivotPatient) {
        cout << i << " ";
    }
    cout << endl;
}

void NurseVrp::printWScore() {
    cout << "Patient w_score: " << endl;
    vector<Score *> temp;
    while (!w_score.empty()) {
        Score *s_temp = w_score.top();
        temp.emplace_back(s_temp);
        cout << "Patient " << s_temp->patient << " w_score: " << s_temp->weight << endl;
        w_score.pop();
    }

    for (auto &s : temp) {
        // cout << "Patient " << (s->patient)->NodeGetName() << " w_score: " << s->weight << endl;
        w_score.emplace(s);
    }
}

void NurseVrp::HeuristicFindPivotCriterion1(set<int> &pivotPatientSet) {
    ConstructEdgeForMaxClique();  // Construct the graph for max clique
    #ifdef DEBUG_GETMAXCLIQUE
    cout << "==> Print Vertex before find max clique: " << endl;
    printVertex();
    #endif
    // Find the max clique
    findCliques();
    #ifdef DEBUG_GETMAXCLIQUE
    cout << "==> Print Vertex after find max clique: " << endl;
    printVertex();
    #endif
    // update pivotPatient, pass the max clique to pivotPatient
    int pivot_size1 = getMaxCliqueNum();
    #ifdef DEBUG_GETMAXCLIQUE
    cout << "==> Check pivot patient when calculate j1_pivot_num and j2_pivot_num: " << endl;
    cout << "==> Pivot size: " << pivot_size1 << endl;
    printMaxClique();
    #endif
    #ifdef DEBUG_GETMAXCLIQUE_ONLY
    printMaxClique();
    #endif
    // update the j1_pivot_num and j2_pivot_num
    /*
    for (auto &p : pivotPatient) {
        #ifdef DEBUG_GETMAXCLIQUE
        cout << "pivot patient: " << p.NodeGetName() << " type: " << p.PatientGetType() << endl;
        #endif
        if (p.PatientGetType() == true){
            j1_pivot_num += 1;
        } else {
            j2_pivot_num += 1;
        }
    }*/

    // update the pivotPatientSet
    for (auto &p: clique) {
        pivotPatientSet.insert(p);
    }
    // the pivotPatientSet is updated contain all the pivot patient satisfy the criterion 1
}

void NurseVrp::print_queue_patient(queue<Patient> q) {
    cout << "==> Print backup patient: " << endl;
    while (!q.empty()){
        Patient tempP = q.front();
        cout << "backup patient: " << tempP.NodeGetName() << " type: " << tempP.PatientGetType() << endl;
        q.pop();
    }
    cout << "==> End of print backup patient." << endl;
}

void NurseVrp::HeuristicFindPivotCriterion2(set<int> &pivotPatientSet) {
    initOpenClinic(); // open all the clinic
    for (auto &p : p1) {  // set weight for all p1
        PatientSetScoreForWeight(p); // w_score is updated
    }
    for (auto &p : p2) {  // set weight for all p2
        PatientSetScoreForWeight(p); // w_score is updated
    }

    int j1_pivot_num = 0;
    int j2_pivot_num = 0;

    queue<int> backupPatient; // not sure yet, maybe I won't use it

    #ifdef DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
    printWScore();
    #endif
    // exit(0);

    // pivotPatient is updated
    // traverse the w_score from the top to the bottom
    while (!w_score.empty()){
        Score *s = w_score.top();
        int p = s->patient;

        // cout << "Patient " << p << " w_score: " << s->weight << endl;

        // consider the patient satisfy the criterion 1
        if (pivotPatientSet.find(p) != pivotPatientSet.end()){
            // p is type 1, and the number of pivot type 1 is less than the number of visiting nurse
            if (p < nump1 && j1_pivot_num < numvr){
                pivotPatient.emplace_back(p);
                // pivotPatientSet.insert(p.NodeGetName());
                j1_pivot_num += 1;
            }
            #ifdef DEBUG_PRINT_PIVOT_BY_PATIENT_TYPE
            cout << "Type 1 Patients (Pivot): " << p << endl;
            #endif
            // p is type 2, and the number of pivot type 2 is less than the number of clinic and hospital nurse
            if (p >= nump1 && j2_pivot_num < numnr + numhr){
                pivotPatient.emplace_back(p);
                // pivotPatientSet.insert(p.NodeGetName());
                j2_pivot_num += 1;
            }
        }
        else {
            // if the patient does not satisfy the criterion 1, then put it into the backupPatient
            // I don't know how to reuse the backPatient.....
            backupPatient.emplace(p);
        }

        // pop the top element
        w_score.pop();


        // stop iff both j1_pivot_num and j2_pivot_num are enough
        // it is possible that the onr of the type of pivot patients are not enough
        // for example, the number of pivot type 1 is less than the number of visiting nurse
        // but the number of pivot type 2 is enough
        // Or, the number of pivot type 1 is enough but
        // the number of pivot type 2 is less than the number of clinic and hospital nurse
        // under the above situation, it wont stop
        // even though the number of pivot patients wit enough number won't increase
        // since the second if statement
        if (j1_pivot_num == numvr && j2_pivot_num == numnr + numhr) {
            break;
        }
    }

    #ifdef DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
    cout << "==> Print pivot patient: " << endl;
    for (auto &p : pivotPatient){
        if (p < nump1){
            cout << "pivot patient: " << p << " type: " << 1 << endl;
        } else {
            cout << "pivot patient: " << p << " type: " << 2 << endl;
        }
    }
    cout << "==> End of print pivot patient." << endl;
    #endif

    // after iterate all the patients in the w_score
    // if the pivot patients are not enough, we need to find more pivot patients from the backupPatient
    // j1_pivot_num = 0
    if (j1_pivot_num < numvr || j2_pivot_num < numnr + numhr){

        #ifdef DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
        print_queue_patient(backupPatient);
        cout << "size of backup patient after print: " << backupPatient.size() << endl;
        #endif

        while (!backupPatient.empty()){
            #ifdef DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
            cout << "get in while loop" << endl;
            #endif
            int tempPIndex = backupPatient.front();
            if (tempPIndex < nump1 && j1_pivot_num < numvr){
                pivotPatient.emplace_back(tempPIndex);
                pivotPatientSet.insert(tempPIndex);
                j1_pivot_num += 1;
            }
            else if (tempPIndex >= nump1 && j2_pivot_num < numnr + numhr){
                pivotPatient.emplace_back(tempPIndex);
                pivotPatientSet.insert(tempPIndex);
                j2_pivot_num += 1;
            }
            backupPatient.pop();

            // stop iff both j1_pivot_num and j2_pivot_num are enough
            if (j1_pivot_num == numvr && j2_pivot_num == numnr + numhr) {
                break;
            }
        }
    }

    // at the end of this algorithm, the pivotPatient and pivotPatientSet is updated
    // the pivotPatientSet is updated contain and the number of pivot patients are enough

    #ifdef DEBUG_IN_HEURISTIC_FINDPIVOTCRITERION2
    cout << "numvr: " << numvr << endl;
    cout << "numnr+numhr: " << numnr + numhr << endl;
    cout << "j1_pivot_num: " << j1_pivot_num << endl;
    cout << "j2_pivot_num: " << j2_pivot_num << endl;
    cout << "pivotPatientSet: " << endl;
    for (auto &p : pivotPatientSet) {
        cout << p << " ";
    }
    cout << endl;
    cout << "pivotPatient: " << endl;
    for (auto &p : pivotPatient) {
        cout << p << " ";
    }
    cout << endl;
    #endif
}

void NurseVrp::HeuristicUpdateNonPivotPatient(set<int> &pivotPatientSet) {
    // update the nonPivotPatient
    for (auto &p : p1) {
        if (pivotPatientSet.find(p.NodeGetName()) == pivotPatientSet.end()) { // if p is not in the pivotPatientSet
            nonPivotPatient.emplace_back(p.NodeGetName());
        }
    }
    for (auto &p : p2) {
        if (pivotPatientSet.find(p.NodeGetName()) == pivotPatientSet.end()) { // if p is not in the pivotPatientSet
            nonPivotPatient.emplace_back(p.NodeGetName());
        }
    }

#ifdef DEBUG_PRINT_NON_PIVOT_PATIENT
    for (auto &p : nonPivotPatient) {
        if (p < nump1) {
            cout << "Type 1 Patient: " << p << " is not pivot patient" << endl;
        } else {
            cout << "Type 2 Patient: " << p << " is not pivot patient" << endl;
        }
    }
#endif
}

void NurseVrp::HeuristicPivotPatient() {
    int j1_pivot_num = 0;
    int j2_pivot_num = 0;

    set<int> pivotPatientSet;

    // Criterion 1: The incompatiblity of patients
    HeuristicFindPivotCriterion1(pivotPatientSet);

    #ifdef DEBUG_PIVOT_CONVERT_CHECK
    printPivot();
    printMaxClique();
    #endif

    // Criterion 2: The score of patients, descending order
    HeuristicFindPivotCriterion2(pivotPatientSet);

    // update the nonPivotPatient
    HeuristicUpdateNonPivotPatient(pivotPatientSet);

    #ifdef DEBUG_PRINT_PIVOT_AFTER_2CRITERION
    printPivot();
    #endif

    #ifdef DEBUG_W_SCORE
    printWScore();
    #endif
}