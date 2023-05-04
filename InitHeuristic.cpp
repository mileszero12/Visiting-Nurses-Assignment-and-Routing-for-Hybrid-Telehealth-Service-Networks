#include "NurseVrp.hpp"
#include <queue>
#include <algorithm>


void NurseVrp::assignPivotToNurseAndClinic() {
    int cur_vr = 0; // for travelling nurse
    int cur_nc = 0; // for clinic nurse
    int cur_nh = 0; // for hospital nurse

    //rank the pivot type 2 patient by their distance from the hospital, ascending order
    priority_queue<DisToHos, vector<DisToHos>, CompareDis> D;
    // new vector only contain type 1 pivot patient
    vector<int> D1;
    for (auto &p: pivotPatient) {
        // cout << "patient : " << p << endl;
        if (p >= nump1) { // only consider type 2 patient here, type 2 patient is assigned to hospital nurse
            double dis = travelTime( c[numc], p2[p - nump1]);
            DisToHos temp(p - nump1, dis);
            D.push(temp);
        } else {
            D1.emplace_back(p);
        }
    }

    // assign the top numhr pivot patient to the hospital nurse
    while (!D.empty() && cur_nh < numhr) {
        DisToHos temp = D.top();
        p2[temp.patient].UpdateLabelList1_Init(p2[temp.patient].PatientGetLabelListLen(), &c[numc], HOSPITAL_ROUTE,
                                                      &hr[cur_nh]);
        // update the cost and starting time
        p2[temp.patient].UpdateLabelList2_Init(p2[temp.patient].PatientGetLabelListLen(),
                                                     &c[numc], hr[cur_nh], false);
        // next index in LabelList
        p2[temp.patient].PatientLabelListIndAdd();

        D.pop();

        // cout << "patient " << temp.patient << " is assigned to hospital nurse " << cur_nh << endl;
        // update the current hospital nurse
        cur_nh++;
    }

    // assign the rest pivot patient to the clinic nurse
    while (!D.empty()) {
        DisToHos temp = D.top();

        // find the nearest clinic
        int nearest_clinic_index;
        double nearest_clinic_dist = WANCHAI;
        for (auto &clinic: c) { // for each clinic
            if (clinic.NodeGetName() < numc) {// only consider the clinic (not the hospital)
                if (clinic.checkNodeIsInAdjList(&p2[temp.patient])) { // if p is in the clinic's adj list
                    if (clinic.getDistance(&p2[temp.patient]) <
                        nearest_clinic_dist) { // if the distance is smaller than the current nearest clinic
                        nearest_clinic_index = clinic.NodeGetName(); // update the nearest clinic index
                        nearest_clinic_dist = clinic.getDistance(&p2[temp.patient]); // update the nearest clinic distance
                    }
                }
            }
        }

        // only update clinic, nurse, and type
        p2[temp.patient].UpdateLabelList1_Init(p2[temp.patient].PatientGetLabelListLen(), &c[nearest_clinic_index], CLINIC_ROUTE, &nr[cur_nc]);
        // update the cost and starting time
        p2[temp.patient].UpdateLabelList2_Init(p2[temp.patient].PatientGetLabelListLen(),&c[nearest_clinic_index], nr[cur_nc], false);
        // cout << "patient " << temp.patient << " is assigned to clinic " << p2[temp.patient].labelList[p2[temp.patient].PatientGetLabelListLen()].getClinic()->NodeGetName() << endl;
        // next index in LabelList
        p2[temp.patient].PatientLabelListIndAdd();

        D.pop();
        // update the current clinic nurse
        cur_nc++;
    }

    // assign the pivot patient type 1 to the travelling nurse
    for (auto &p: D1) {
        // find the nearest clinic
        int nearest_clinic_index;
        double nearest_clinic_dist = WANCHAI;
        for (auto &clinic: c) { // for each clinic
            if (clinic.checkNodeIsInAdjList(&p1[p])) { // if p is in the clinic's adj list
                if (clinic.getDistance(&p1[p]) <
                    nearest_clinic_dist) { // if the distance is smaller than the current nearest clinic
                    nearest_clinic_index = clinic.NodeGetName(); // update the nearest clinic index
                    nearest_clinic_dist = clinic.getDistance(&p1[p]); // update the nearest clinic distance
                }
            }
        }

        #ifdef DEBUG_PRINT_PIVOT_LABEL_LIST
        cout << "Pivot patient " << p1[p].NodeGetName() << " is assigned to clinic " << nearest_clinic.NodeGetName() << endl;
        #endif

        // only update clinic, nurse, and type
        p1[p].UpdateLabelList1_Init(p1[p].PatientGetLabelListLen(), &c[nearest_clinic_index], VISITING_ROUTE, &vr[cur_vr]);
        // update the cost and starting time
        p1[p].UpdateLabelList2_Init(p1[p].PatientGetLabelListLen(), &c[nearest_clinic_index],vr[cur_vr], false);
        // next index in LabelList
        p1[p].PatientLabelListIndAdd();
        #ifdef DEBUG_PRINT_PIVOT_LABEL_LIST
        cout << "Label List len: " << p1[p].PatientGetLabelListLen() << endl;
        #endif
        // next nurse
        cur_vr++;
    }

    #ifdef DEBUG_PRINT_PIVOT_LABEL_LIST
    for (auto &p : pivotPatient){
        if (p >= nump1) // only consider type 2 patient here
            p2[p - nump1].printLabelList();
        else{
            p1[p].printLabelList();
        }
    }
    #endif
}

void NurseVrp::assignNonPivotToNurseAndClinic() {
    for (auto &i: nonPivotPatient) {
        if (i >= nump1) { // only consider type 2 patient here
            currentP_AddLabel(p2[i - nump1]);
            // #ifdef DEBUG_CHECK_PARENT
            // cout << "Patient " << p1[i].NodeGetName() << "'s Parent in currentP_AddLabel() is  " << p2[i - nump1].labelList[p2[i - nump1].PatientGetLabelListLen() - 1].getParent() << endl;
            // #endif

            #ifdef DEBUG_PRINT_NON_PIVOT_LABEL_LIST
            p2[i-nump1].printLabelList();
            #endif
        } else {
            currentP_AddLabel(p1[i]);
            // #ifdef DEBUG_CHECK_PARENT
            // cout << "Patient " << p1[i].NodeGetName() << "'s Parent in currentP_AddLabel() is  " << p1[i].labelList[p1[i].PatientGetLabelListLen() - 1].getParent() << endl;
            // #endif

            #ifdef DEBUG_PRINT_NON_PIVOT_LABEL_LIST
            p1[i].printLabelList();
            #endif
        }
    }

    #ifdef DEBUG_PRINT_NON_PIVOT_LABEL_LIST
    cout << "===== out of the loop:" << endl;
    for (int i = 0; i < nump; i++){
        if (i >= nump1) // only consider type 2 patient here
            p2[i - nump1].printLabelList();
        else{
            p1[i].printLabelList();
        }
    }
    #endif
}


void NurseVrp::currentP_AddLabel(Patient &p) {
    if (p.PatientGetType() == 1) { // if the patient is type 1, it can be visited only by the travelling route
        #ifdef DEBUG_ADD_LABEL
        cout << "==========================================================" << endl;
        cout << "DEBUG_ADD_LABEL: currentP_AddLabel() is called for type 1 patient " << p.NodeGetName() << endl;
        #endif

        for (auto &h: pivotPatient) { // for all the type 1 patient, no need to check the type of the labels
            // if the current patient p is in the adj list of the patient j, if current patient p is not the same as the patient j
            Patient* j;
            if (h < nump1) {
                j = &p1[h];
            } else {
                j = &p2[h - nump1];
            }

            #ifdef DEBUG_ADD_LABEL
            cout << "Find possible pivot source:  " << endl;
            #endif

            if (j->NodeGetName() != p.NodeGetName() && j->checkNodeIsInAdjList(&p)) {
                #ifdef DEBUG_ADD_LABEL
                cout << "    Patient " << j->NodeGetName() << " is a possible source." << endl;
                cout << "    The length of " << j->NodeGetName() << "'s LabelList is " << j->PatientGetLabelListLen() << endl;
                #endif
                for (int i = 0; i < j->PatientGetLabelListLen(); i++) { // for all the labels of the patient j
                    // if the current label is valid, j start time + j service time + j -> p distance <= p end time
                    //#ifdef DEBUG_ADD_LABEL
                    // cout << "      Label " << i << " is possible to be extended." << endl;
                    // #endif
                    if (j->labelList[i].getST() + j->PatientGetServiceTime() + j->travelTime(p) <= p.PatientGetEndTime()) {
                        UpdateLabelList(p, j, p.PatientGetLabelListLen(), VISITING_ROUTE, &(j->labelList[i]), true);
                        #ifdef DEBUG_CHECK_PARENT
                        cout << "      Label " << i << " can be extended, add this label to the LabelList in type 1 patient " << p.NodeGetName() << ", and current LabelList index is " << p.PatientGetLabelListLen() - 1 << endl;
                        cout << "      Update Parent for " << p.NodeGetName() << " and its parent is " << p.labelList[p.PatientGetLabelListLen() - 1].getParent() << endl;
                        #endif
                    }
                }
            }
        }

        #ifdef DEBUG_ADD_LABEL
        cout << "Check updated type 1 patient " << p.NodeGetName() << "'s LabelList length. " << endl;
        cout << "Patient " << p.NodeGetName() << " label length: " << p.PatientGetLabelListLen() << endl;
        cout << "==========================================================" << endl;
        #endif
    }
    else { // if the patient is type 2
        #ifdef DEBUG_ADD_LABEL
        cout << "==========================================================" << endl;
        cout << "DEBUG_ADD_LABEL: currentP_AddLabel() is called for type 2 patient " << p.NodeGetName() << endl;
        #endif

        #ifdef DEBUG_ADD_LABEL
        cout << "Find possible pivot source:  " << endl;
        #endif
        for (auto &h: pivotPatient) { // from type 2 patient
            // if the current patient p is in the adj list of the patient j, if current patient p is not the same as the patient j
            Patient* j;
            if (h < nump1) {
                j = &p1[h];
            } else {
                j = &p2[h - nump1];
            }



            if (j->NodeGetName() != p.NodeGetName() && j->checkNodeIsInAdjList(&p)) {
                #ifdef DEBUG_ADD_LABEL
                cout << "    Patient " << j->NodeGetName() << " is a possible source." << endl;
                cout << "    The length of " << j->NodeGetName() << "'s LabelList is " << j->PatientGetLabelListLen() << endl;
                #endif
                for (int i = 0; i < j->PatientGetLabelListLen(); i++) { // for all the labels of the patient j
                    // clinic route, if the current label is valid, j start time + j service time + j -> p distance <= p end time
                    // visiting route, if the current label is valid, j start time + j service time + j -> p distance <= p end time
                    // #ifdef DEBUG_ADD_LABEL
                    // cout << "Label " << i << " can be extended." << endl;
                    // #endif
                    if (j->labelList[i].getRouteType() == VISITING_ROUTE) { // visiting route
                        if (j->labelList[i].getST() + j->PatientGetServiceTime() + j->travelTime(p) <=
                            p.PatientGetEndTime()) {
                            UpdateLabelList(p, j, p.PatientGetLabelListLen(), VISITING_ROUTE, &(j->labelList[i]), true);
                            #ifdef DEBUG_CHECK_PARENT
                            cout << "      Label " << i << " can be extended, add this label to the LabelList in type 1 patient " << p.NodeGetName() << ", and current LabelList index is " << p.PatientGetLabelListLen() - 1 << endl;
                            cout << "      Update Parent for " << p.NodeGetName() << " and its parent is " << p.labelList[p.PatientGetLabelListLen() - 1].getParent() << endl;
                            #endif
                        }
                    }
                    else { // clinic and hospital route
                        if (j->labelList[i].getST() + j->PatientGetServiceTime() <= p.PatientGetEndTime()) {
                            UpdateLabelList(p, j, p.PatientGetLabelListLen(), j->labelList[i].getRouteType(), &(j->labelList[i]), true);
                            #ifdef DEBUG_CHECK_PARENT
                            cout << "      Label " << i << " can be extended, add this label to the LabelList in type 1 patient " << p.NodeGetName() << ", and current LabelList index is " << p.PatientGetLabelListLen() - 1 << endl;
                            cout << "      Update Parent for " << p.NodeGetName() << " and its parent is " << p.labelList[p.PatientGetLabelListLen() - 1].getParent() << endl;
                            #endif
                        }
                    }
                }
            }
        }
        /*
        for (auto &j: p1) { // from type 1 patient
            // if the current patient p is in the adj list of the patient j, if current patient p is not the same as the patient j
            if (j.NodeGetName() != p.NodeGetName() && j.checkNodeIsInAdjList(&p)) {
                #ifdef DEBUG_ADD_LABEL
                cout << "Patient " << j.NodeGetName() << " is a possible source." << endl;
                #endif
                for (int i = 0; i < j.PatientGetLabelListLen(); i++) { // for all the labels of the patient j
                    // clinic route, if the current label is valid, j start time + j service time + j -> p distance <= p end time
                    // visiting route, if the current label is valid, j start time + j service time + j -> p distance <= p end time
                    // #ifdef DEBUG_ADD_LABEL
                    // cout << "Label " << i << " can be extended." << endl;
                    // #endif
                    if (j.labelList[i].getST() + j.PatientGetServiceTime() + j.travelTime(p) <= p.PatientGetEndTime()) {
                        UpdateLabelList(p, j, p.PatientGetLabelListLen(), j.labelList[i].getRouteType(),
                                              &j.labelList[i], true);
                        #ifdef DEBUG_CHECK_PARENT
                        cout << "      Label " << i << " can be extended, add this label to the LabelList in type 1 patient " << p.NodeGetName() << ", and current LabelList index is " << p.PatientGetLabelListLen() - 1 << endl;
                        cout << "      Update Parent for " << p.NodeGetName() << " and its parent is " << p.labelList[p.PatientGetLabelListLen() - 1].getParent() << endl;
                        #endif
                    }
                }
            }
        }
         */
        #ifdef DEBUG_ADD_LABEL
        cout << "Check updated type 2 patient " << p.NodeGetName() << "'s LabelList length. " << endl;
        cout << "Patient " << p.NodeGetName() << " label length: " << p.PatientGetLabelListLen() << endl;
        cout << "==========================================================" << endl;
        cout << endl;
        #endif

        #ifdef DEBUG_PRINT_NON_PIVOT_LABEL_LIST2
        cout << "================== Print Label List ====================" << endl;
        p.printLabelList();
        cout << "========================================================" << endl;
        #endif
    }

}

void NurseVrp::ConvertLabelToRoute() {
    // there are 3 sequence:
    // "Seq" is the sequence record the route and clinic and nurse information (int)
    // "tempSeq" is the sequence record the reversed route, will be updated in the inner for loop. (Each Label)
    // "Seq_len" is the sequence record the length of each route (int)
    // Seq_obj, Seq_vval, Seq_vind, Seq_vbeg for the constriant matrix

    int SeqIndex = 0; // for Seq
    int vval_index = 0; // for Seq_vval
    int SeqIndexBegin = 0; // for Seq_vbeg


    for (auto &i: p1) {
        updateRoute(SeqIndex, vval_index, SeqIndexBegin, &i);
    }
    for (auto &i: p2) {
        updateRoute(SeqIndex, vval_index, SeqIndexBegin, &i);
    }

    #ifdef CHECK_ROUTE_IN_LABEL_TO_ROUTE
    printRoute();
    printVBeg();
    #endif

}



void NurseVrp::updateRoute(int &SeqIndex, int &vval_index, int &SeqIndexBegin, Patient *p) {
    #ifdef DEBUG_UPDATE_ROUTE
    cout << "======================== updateRoute ==================================" << endl;
    cout << "Patient " << p->NodeGetName() << " in updateRoute() function:" << endl;
    #endif

    for (int j = 0; j < p->PatientGetLabelListLen(); j++) { // traversal all the labels of the patient p
        auto templable = &(p->labelList[j]);
        //cout << templable->getEndVex() << endl;

        #ifdef DEBUG_UPDATE_ROUTE
        cout << "=> Patient " << p->NodeGetName() << ", label " << j << " : " << endl;
        #endif

        int templen = 0; // the length of the route
        vector<int> tempSeq;
        bool flag = true; // if the route is valid

        // since the route we get is reversed, we need to reverse it
        while (flag){
            #ifdef DEBUG_UPDATE_ROUTE
            cout << "Patient " << templable->getEndVex() << " <- ";
            // cout << endl;
            #endif

            tempSeq.emplace_back(templable->getEndVex());  // add patient to the temp seq
            templen += 1; // update the length of the "route"

            if (templable->getParentLabel() == nullptr) {
                flag = false;
            }
            else{
                templable = templable->getParentLabel();
            }
        }

        #ifdef DEBUG_UPDATE_ROUTE
        cout << endl;
        cout << "    Check tempSeq: ";
        for (int & i : tempSeq) {
            cout << i << " ";
        }
        cout << endl;
        #endif


        // update "Seq", reverse the temp seq
        // Seq = reversed route + clinic + route type + nurse
        for (auto it = tempSeq.rbegin(); it != tempSeq.rend(); ++it) {
            //cout << "***********" << *it << " SeqIndex: " << SeqIndex << endl;
            Seq[SeqIndex] = *it;
            SeqIndex += 1;
        }
        //cout << "out of the loop " << SeqIndex << endl;
        Seq[SeqIndex] = templable->getClinic()->NodeGetName();  // add clinic to the seq
        SeqIndex += 1;
        Seq[SeqIndex] = templable->getRouteType(); // add route type to the seq
        SeqIndex += 1;
        Seq[SeqIndex] = templable->getNurse()->NurseGetName(); // add Nurse to the seq
        SeqIndex += 1;

        // printRoute();

        // sort the temp route then it will be easy to add it to the constraint matrix, the order
        // sort(tempSeq.begin(), tempSeq.end());

        // update the complete cost for the route matrix
        double cur_cost = templable->getCurCost();
        if (templable->getRouteType() == VISITING_ROUTE) {
            cur_cost += travelCost(templable->getClinic(), *p);
        } else {
            cur_cost += travelCostPatient(templable->getClinic(), *p);
        }


        Seq_obj[n_route] = cur_cost; // update cost
        Seq_vlen[n_route] = templen; // update the length of the route
        Seq_vbeg[n_route] = SeqIndexBegin; // update the beginning non-zero index of the route

        // calculate vval, vind, vbeg
        for (auto it: tempSeq) { // route
                Seq_vval[vval_index] = 1; // update patient val
                Seq_vind[vval_index] = it; // update patient constraint matrix index
                vval_index += 1;
        }

        Seq_vval[vval_index] = 1; // update nurse val
        if (templable->getRouteType() == VISITING_ROUTE){ // route type
            Seq_vind[vval_index] = nump + templable->getNurse()->NurseGetName(); // nurse constraint matrix index
            #ifdef DEBUG_UPDATE_ROUTE_NURSE_CHECK_VIND
            cout << "visiting tempNurse Index: " << Seq_vind[vval_index] - nump << endl;
            #endif
        }
        else if (templable->getRouteType() == CLINIC_ROUTE){ // nurse num
            Seq_vind[vval_index] = nump + numvr + templable->getNurse()->NurseGetName();
            #ifdef DEBUG_UPDATE_ROUTE_NURSE_CHECK_VIND
            cout << "clinic tempNurse Index: " << Seq_vind[vval_index] - nump - numvr << endl;
            #endif
        }
        else{
            Seq_vind[vval_index] = nump + numvr + numhr + templable->getNurse()->NurseGetName();
            #ifdef DEBUG_UPDATE_ROUTE_NURSE_CHECK_VIND
            cout << "hospital tempNurse Index: " << Seq_vind[vval_index] -nump - numvr - numnr << endl;
            #endif
        }


        vval_index += 1;
        n_route += 1; // next route
        SeqIndexBegin += templen + 1; // update the beginning non-zero index of the route
    }
}


void NurseVrp::printRoute() {
    cout << "======================== PrintRoute ==================================" << endl;
    int count = 0;
    for (int i = 0; i < n_route; i++) {
        cout << "Route " << i << " has " << Seq_vlen[i] << " element. " << endl;
        cout << "Patient: ";
        for (int j = 0; j < Seq_vlen[i]; j++) {
            cout << Seq[j + count] << " ";
        }
        cout << endl;
        count += Seq_vlen[i];
        cout << "clinic: " << Seq[count] << endl;
        count += 1;
        cout << "route type: " << Seq[count] << endl;
        count += 1;
        cout << "nurse: " << Seq[count] << endl;
        count += 1;
        cout << "--------------------------------------------------------------" << endl;
    }
    cout << "==========================================================" << endl;
}

void NurseVrp::printVLen(){
    for (int i = 0; i < n_route; i++) {
        cout << i << " route's length: " << Seq_vlen[i] << endl;
    }
}

void NurseVrp::printVBeg(){
    int curInd = 0;
    for (int i = 0; i < n_route; i++) {
        cout << i << " route's beg: " << Seq_vbeg[i] << ", ";
        cout << "the number of nonzero elements: " << Seq_vlen[i] + 1 << endl;
        for (int j = 0; j < Seq_vlen[i] + 1; j++) {
            int newj = curInd + j;
            if (j < Seq_vlen[i]){
                cout << "    the nonzero element is in patient constraint: " << Seq_vind[newj] << endl;
            }
            else{
                if (Seq_vind[newj] < nump + numvr && Seq_vind[newj] >= nump){
                    cout << "    the nonzero element is in visiting nurse constraint: " << Seq_vind[newj] - nump << endl;
                }
                else if (Seq_vind[newj] >= nump + numvr && Seq_vind[newj] < nump + numvr + numnr){
                    cout << "    the nonzero element is in clinic nurse constraint: " << Seq_vind[newj] - nump - numvr << endl;
                }
                else{
                    cout << "    the nonzero element is in hospital nurse constraint: " << Seq_vind[newj]  - nump - numvr -numnr << endl;
                }
            }
        }
        curInd += Seq_vlen[i] + 1;
    }
}

void NurseVrp::printObj(){
    for (int i = 0; i < n_route; i++) {
        cout << i << " route's obj: " << Seq_obj[i] << endl;
    }
}