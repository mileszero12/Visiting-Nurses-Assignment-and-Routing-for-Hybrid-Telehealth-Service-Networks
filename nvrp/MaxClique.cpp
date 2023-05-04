#include "NurseVrp.hpp"
#include <fstream>
#include <random>
#include <algorithm>


auto NurseVrp::select_random() {
    size_t n;
    n = rand() % vertexList.size();

#ifdef DEBUG_CLIQUES
    cout << "select_random function: vertexList length:" << vertexList.size() << endl;
    cout << "select_random function: n = " << n << endl;
#endif

    auto it = begin(vertexList);
    advance(it, n);

#ifdef DEBUG_CLIQUES
    cout << "select_random function, final output is: " << *it << endl;
#endif
    return it;
}

void NurseVrp::findCliques() {
#ifdef DEBUG_CHECK_VECTOR_LIST
    cout << "findCliques function" << endl;
#endif

    int rand = *select_random();

#ifdef DEBUG_CHECK_VECTOR_LIST
    cout << "rand = " << rand << endl;
#endif

#ifdef DEBUG_CLIQUES
    cout << "findCliques() function, randon select vertex is: " << rand << endl;
#endif
    clique.emplace_back(rand);

    for (auto &v: vertexList) {
#ifdef DEBUG_CLIQUES
        cout << "findCliques() function, for each v: " << v << endl;
#endif
        // cout << "v: " << v << endl;
        if (find(clique.begin(), clique.end(), v) != clique.end()) { // if v is in clique

#ifdef DEBUG_CLIQUES
            cout << "findCliques() function: " << v << " is in clique" << endl;
#endif

            continue;
        }
        bool isNext = true;
        for (auto &cli: clique) { // for each vertex c in clique
            if (graph[v][cli] == 1) {
                // cout << "graph[v][c]= " << graph[v][c] << endl;
                // cout << "v: " << v << " c: " << c << endl;
#ifdef DEBUG_CLIQUES
                cout << "findCliques() function, Connected v: " << v << " c: " << c << endl;
#endif

                continue;
            } else {
                isNext = false;
                break;
            }
        }
        if (isNext) {
            clique.emplace_back(v);
        }
    }
}

void NurseVrp::ConstructEdgeForMaxClique() {
    int temp = 0;
    // For J1
    for (int j = 0; j < nump1; ++j) {
        for (int j2 = 0; j2 < nump1; ++j2) {
#ifdef MAX_CLIQUE_PIVOT_CHECK
            cout << "i = " << p1[j].NodeGetName() << ", j = " << p1[j2].NodeGetName() << endl;
            cout << "E_i + T_ij = " << p1[j].PatientGetStartTime() + travelTime(p1[j], p1[j2]) << endl;
            cout << "L_j = " << p1[j2].PatientGetEndTime() << endl;
            cout << "E_j + T_ji = " << p1[j2].PatientGetStartTime() + travelTime(p1[j], p1[j2]) << endl;
            cout << "L_i = " << p1[j].PatientGetEndTime() << endl;
#endif
            if (p1[j].PatientGetStartTime() + travelTime(p1[j], p1[j2]) >
                p1[j2].PatientGetEndTime() &&
                p1[j2].PatientGetStartTime() + travelTime(p1[j], p1[j2]) >
                p1[j].PatientGetEndTime() && j != j2) {
                edges.emplace_back(p1[j].NodeGetName(), p1[j2].NodeGetName());
                vertexList.insert(p1[j].NodeGetName());
                vertexList.insert(p1[j2].NodeGetName());
                temp++;
#ifdef DEBUG_FIND_EDGE
                cout << "J1 to J1: " << p1[j].NodeGetName() << " " << p1[j2].NodeGetName() << endl;
#endif
            }
        }
        /*
        for (int j2 = 0; j2 + nump1 < nump; ++j2) {
            if (p1[j].PatientGetStartTime() + travelTime(p1[j], p2[j2]) > p2[j2].PatientGetEndTime() && p2[j2].PatientGetStartTime() + travelTime(p1[j], p2[j2]) > p1[j].PatientGetEndTime()) {
                edges.emplace_back(make_tuple(p1[j].NodeGetName(), p2[j2].NodeGetName()));
                vertexList.insert(p1[j].NodeGetName());
                vertexList.insert(p2[j2].NodeGetName());
                temp++;
                #ifdef DEBUG_FIND_EDGE
                cout << "J1 to J2: " << p1[j].NodeGetName() << " " << p2[j2].NodeGetName() << endl;
                #endif
            }
        }
         */
    }

    // For J2
    for (int j = 0; j + nump1 < nump; ++j) {
        /*
        for (int j2 = 0; j2 < nump1; ++j2){
            #ifdef MAX_CLIQUE_PIVOT_CHECK
            cout << "i = " << p2[j].NodeGetName() << ", j = " << p1[j2].NodeGetName() << endl;
            cout << "E_i + T_ij = " << p2[j].PatientGetStartTime() + travelTime(p2[j], p1[j2]) << endl;
            cout << "L_j = " << p1[j2].PatientGetEndTime() << endl;
            cout << "E_j + T_ji = " << p1[j2].PatientGetStartTime() + travelTime(p2[j], p1[j2]) << endl;
            cout << "L_i = " << p2[j].PatientGetEndTime() << endl;
            #endif

            if (p2[j].PatientGetStartTime() + travelTime(p2[j], p1[j2]) > p1[j2].PatientGetEndTime() && p1[j2].PatientGetStartTime() + travelTime(p2[j], p1[j2]) > p2[j].PatientGetEndTime() && j != j2) {
                edges.emplace_back(make_tuple(p2[j].NodeGetName(), p1[j2].NodeGetName()));
                vertexList.insert(p2[j].NodeGetName());
                vertexList.insert(p1[j2].NodeGetName());
                temp++;
                #ifdef DEBUG_FIND_EDGE
                cout << "J2 to J1: " << p2[j].NodeGetName() << " " << p1[j2].NodeGetName() << endl;
                #endif
            }
        }
        */

        for (int j2 = 0; j2 + nump1 < nump; ++j2) {
#ifdef MAX_CLIQUE_PIVOT_CHECK
            cout << "i = " << p2[j].NodeGetName() << ", j = " << p2[j2].NodeGetName() << endl;
            cout << "E_i + 0 = " << p2[j].PatientGetStartTime()<< endl;
            cout << "L_j = " << p2[j2].PatientGetEndTime() << endl;
            cout << "E_j + 0 = " << p2[j2].PatientGetStartTime() << endl;
            cout << "L_i = " << p2[j].PatientGetEndTime() << endl;
#endif
            if (p2[j].PatientGetStartTime() < p2[j2].PatientGetStartTime() &&
                p2[j2].PatientGetEndTime() > p2[j].PatientGetEndTime() && j != j2) {
                edges.emplace_back(p2[j].NodeGetName(), p2[j2].NodeGetName());
                vertexList.insert(p2[j].NodeGetName());
                vertexList.insert(p2[j2].NodeGetName());
                temp++;
#ifdef DEBUG_FIND_EDGE
                cout << "J2 to J2: " << p2[j].NodeGetName() << " " << p2[j2].NodeGetName() << endl;
#endif
            }
        }
    }
    edgeNum = temp;

    // Construct Graph
    for (int i = 0; i < edgeNum; i++) {
        graph[get<0>(edges[i])][get<1>(edges[i])] = 1;
        graph[get<1>(edges[i])][get<0>(edges[i])] = 1;
    }
}

void NurseVrp::printEdge() {
    for (int i = 0; i < edgeNum; ++i) {
        cout << get<0>(edges[i]) << " " << get<1>(edges[i]) << endl;
    }
}

int NurseVrp::getMaxCliqueNum() {
    return int(clique.size());
}

void NurseVrp::printMaxClique() {
    cout << "Max clique is: " << endl;
    for (int i : clique) {
        cout << i << " ";
    }
    cout << endl;
}

void NurseVrp::printVertex() {
    cout << "The vertex for the clique is: " << endl;
    for (int it : vertexList) {
        cout << it << " ";
    }
    cout << endl;
}

void NurseVrp::cliqueToPivot() {
    cout << clique.size() << endl;
    for (int i : clique) {
        if (i < nump1) {
            pivotPatient.emplace_back(p1[i].NodeGetName());
        } else {
            pivotPatient.emplace_back(p2[i - nump1].NodeGetName());
        }
    }
}