#include <iostream>
#include <vector>
#include <random>
#include <ostream>
#include <bitset>
#include <fstream>
#include <string>
//#include "cereal/archives/binary.hpp"

using namespace std;
const int ratio = 3;

const int numc = 11; // 6 + 1
const int nump1 = 120;
const int nump2 = 80;
const int numvr = 120;
const int numnr = 40;
const int numhr = 40;

class node{
public:
    int name;
    int locationx;
    int locationy;
};

/* patient: name, kind, x, y, service time, starting time, end time */
class patient: public node{
public:
    bool kind;
    double servicetime;
    double start;
    double end;
    vector<patient> adj;
    double dual;
    void set_dual(double d){
        dual = d;
    }
    void set(int na, bool ki, int x, int y, double ser, double sta, double ed){
        name = na;
        kind = ki;
        locationx = x;
        locationy = y;
        servicetime = ser;
        start = sta;
        end = ed;
    }
    bool isPatient(void){
        return true;
    }
};

// clinic 0 is hospital!!!!!!!!!!!
class clinic: public node{
public:
    int setupfee;
    bool open = false;
    vector<patient> adj;
    bool isPatient(void){
        return false;
    }
    void set(int na, int x, int y, int fee, bool op){
        name = na;
        locationx = x;
        locationy = y;
        setupfee = fee;
        open = op;
    }
};

class nurse{
public:
    int name;
    int type;
    double salary;
    double dual;
    void set_dual(double d){
        dual = d;
    }
    bool isPatient(void){
        return false;
    }
    void set(int na, int typ, double sa){
        name = na;
        type = typ;
        salary = sa;
    }
};

vector<clinic> generateClinic(int numc){
    vector<clinic> resc;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrx(0, 200);
    uniform_int_distribution<> distry(0, 200);
    uniform_int_distribution<> distrfee(300, 450); // setup fee

    for (int i = 0; i < numc-1; i++) {
        int tempx = distrx(gen);
        int tempy = distry(gen);
        int fee = distrfee(gen);
        clinic tempc;
        tempc.set(i, tempx, tempy, fee, 0);
        resc.push_back(tempc);
    }

    clinic tempc;
    tempc.set(numc - 1, 100, 100, 0, 1);
    resc.push_back(tempc);


    return resc;
}

double round2(double var){
    double value = (int)(var * 10);
    return (double)value / 10;
}

double round3(double var){
    double value = (int)(var * 100);
    return (double)value / 100;
}

double travelCost(node p1, node p2){
    double xsquare = pow(p1.locationx - p2.locationx, 2);
    double ysquare = pow(p1.locationy - p2.locationy, 2);
    double res = sqrt(xsquare + ysquare) * 0.2;
    return res;
}


double travelDis(node p1, node p2){
    double xsquare = pow(p1.locationx - p2.locationx, 2);
    double ysquare = pow(p1.locationy - p2.locationy, 2);
    double res = sqrt(xsquare + ysquare);
    return res;
}


double travelCostPatient(node p1, node p2){
    double xsquare = pow(p1.locationx - p2.locationx, 2);
    double ysquare = pow(p1.locationy - p2.locationy, 2);
    double res = sqrt(xsquare + ysquare) * 0.36;
    return res;
}


double travelTime(node p1, node p2){
    double xsquare = pow(p1.locationx - p2.locationx, 2);
    double ysquare = pow(p1.locationy - p2.locationy, 2);
    double res = sqrt(xsquare + ysquare) * 0.02;
    return res;
}

// number of patient, clinic set, type 1(0 must at home) or type 2(1), FL(0) or GA(1)
vector<patient> generatePatient(int num1, int numformer, vector<clinic> clinicSet, bool type, bool t2){
    vector<patient> resP;
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> distrx(0, 230);
    uniform_int_distribution<> distry(0, 230);
    uniform_real_distribution<> distrser(0.6, 1); // service time
    for (int i = 0; i < num1; i++) {
        double tempser = round3(distrser(gen));
        int tempx = distrx(gen);
        int tempy = distry(gen);
        patient tempp;
        if (type == 1){
            tempp.set(numformer + i, type, tempx, tempy, tempser, 0, 12);
        }
        else{
            tempp.set(i, type, tempx, tempy, tempser, 0, 12);
        }
        double mindis = 0;
        if (t2 == true) {
            for (int j = 1; j < clinicSet.size(); j++){
                double tempdis = travelTime(tempp, clinicSet[j]);
                if (tempdis < mindis) {
                    mindis = tempdis;
                }
            }
        }
        uniform_real_distribution<> distrstart(mindis, 12 - tempser - 0.4);
        //uniform_real_distribution<> distred(1.5, 2.5);
        uniform_real_distribution<> distred(1.5, 4);
        double tempst = round3(distrstart(gen));
        double inter = round3(distred(gen));
        double temped;
        if (inter > tempser + 0.4) {
            temped = tempst + round3(distred(gen));
        }
        else{
            temped = tempst + tempser + 0.4;
        }

        if (temped > 12){
            temped = 12;
        }
        if (type == 1){
            tempp.set(numformer + i, type, tempx, tempy, tempser, tempst, temped);
        }
        else{
            tempp.set(i, type, tempx, tempy, tempser, tempst, temped);
        }
        resP.push_back(tempp);
    }

    return resP;
}


// number of nurse, visiting nurse (0) or clinic nurse (1) or hospital nurse (2)
vector<nurse> generateNurse(int numn, int type){
    vector<nurse> resn;
    random_device rd;
    mt19937 gen(rd());
    if (type == 0){
        normal_distribution<double> distrsa(296, 24); // mean 37.59
        for (int i = 0; i < numn; i ++){
            double tempsa = round3(distrsa(gen));
            nurse tempn;
            tempn.set(i, type, tempsa);
            resn.push_back(tempn);
        }
    }
    else if (type == 1){
        normal_distribution<double> distrsa(280, 24); // mean 40.88
        for (int i = 0; i < numn; i ++){
            double tempsa = round3(distrsa(gen));
            nurse tempn;
            tempn.set(i, type, tempsa);
            resn.push_back(tempn);
        }
    }
    else{
        normal_distribution<double> distrsa(320, 24); // mean 35.51
        for (int i = 0; i < numn; i ++){
            double tempsa = round3(distrsa(gen));
            nurse tempn;
            tempn.set(i, type, tempsa);
            resn.push_back(tempn);
        }
    }
    return resn;
}

class Data{
public:
    vector<clinic> c;
    vector<patient> p1;
    vector<patient> p2;
    vector<nurse> vr;
    vector<nurse> nr;
    vector<nurse> hr;

};

void generateData(int numc, int nump1, int nump2, int numvr, int numnr, int numhr, string filenum){
    Data data;
    data.c = generateClinic(numc);
    data.p1 = generatePatient(nump1, nump1, data.c, 0, false);
    data.p2 = generatePatient(nump2, nump1, data.c, 1, true);
    data.vr = generateNurse(numvr, 0);
    data.nr = generateNurse(numnr, 1);
    data.hr = generateNurse(numhr, 2);

    ofstream ofile;
    ofile.open("/Users/pikay/Documents/NurseData/original/data/" + to_string(nump1+nump2) + "/" + filenum + ".txt", ios::app); //app is append which means it will put the text at the end

    ofile << numc << endl;
    ofile << nump1 << endl;
    ofile << nump2 << endl;
    ofile << numvr << endl;
    ofile << numnr << endl;
    ofile << numhr << endl;

    for (int i = 0; i < data.c.size(); i ++){
        ofile << i << " " << data.c[i].locationx << " " << data.c[i].locationy << " " << data.c[i].setupfee << std::endl;
    }

    for (int i = 0; i < data.vr.size(); i ++){
        ofile << data.vr[i].name << " " << data.vr[i].type << " " << data.vr[i].salary << endl;
    }
    for (int i = 0; i < data.nr.size(); i ++){
        ofile << data.nr[i].name << " " << data.nr[i].type << " " << data.nr[i].salary << endl;
    }
    for (int i = 0; i < data.hr.size(); i ++){
        ofile << data.hr[i].name << " " << data.hr[i].type << " " << data.hr[i].salary << endl;
    }

    for (int i = 0; i < data.p1.size(); i ++){
        ofile << data.p1[i].name << " " << data.p1[i].kind << " " << data.p1[i].locationx << " " << data.p1[i].locationy << " " << data.p1[i].servicetime << " " << data.p1[i].start << " " << data.p1[i].end << endl;
    }

    for (int i = 0; i < data.p2.size(); i ++){
        ofile << data.p2[i].name << " " << data.p2[i].kind << " " << data.p2[i].locationx << " " << data.p2[i].locationy << " " << data.p2[i].servicetime << " " << data.p2[i].start << " " << data.p2[i].end << endl;
    }
    ofile.close();

    ofstream ofile2;
    ofile2.open("/Users/pikay/Documents/NurseData/highSetup/data/" + to_string(nump1+nump2) + "/" + filenum + ".txt", ios::app); //app is append which means it will put the text at the end

    ofile2 << numc << endl;
    ofile2 << nump1 << endl;
    ofile2 << nump2 << endl;
    ofile2 << numvr << endl;
    ofile2 << numnr << endl;
    ofile2 << numhr << endl;

    for (int i = 0; i < data.c.size(); i ++){
        ofile2 << i << " " << data.c[i].locationx << " " << data.c[i].locationy << " " << data.c[i].setupfee * ratio << std::endl;
    }

    for (int i = 0; i < data.vr.size(); i ++){
        ofile2 << data.vr[i].name << " " << data.vr[i].type << " " << data.vr[i].salary << endl;
    }
    for (int i = 0; i < data.nr.size(); i ++){
        ofile2 << data.nr[i].name << " " << data.nr[i].type << " " << data.nr[i].salary << endl;
    }
    for (int i = 0; i < data.hr.size(); i ++){
        ofile2 << data.hr[i].name << " " << data.hr[i].type << " " << data.hr[i].salary << endl;
    }

    for (int i = 0; i < data.p1.size(); i ++){
        ofile2 << data.p1[i].name << " " << data.p1[i].kind << " " << data.p1[i].locationx << " " << data.p1[i].locationy << " " << data.p1[i].servicetime << " " << data.p1[i].start << " " << data.p1[i].end << endl;
    }

    for (int i = 0; i < data.p2.size(); i ++){
        ofile2 << data.p2[i].name << " " << data.p2[i].kind << " " << data.p2[i].locationx << " " << data.p2[i].locationy << " " << data.p2[i].servicetime << " " << data.p2[i].start << " " << data.p2[i].end << endl;
    }
    ofile2.close();


    ofstream ofile3;
    ofile3.open("/Users/pikay/Documents/NurseData/largeTW/data/" + to_string(nump1+nump2) + "/" + filenum + ".txt", ios::app); //app is append which means it will put the text at the end

    ofile3 << numc << endl;
    ofile3 << nump1 << endl;
    ofile3 << nump2 << endl;
    ofile3 << numvr << endl;
    ofile3 << numnr << endl;
    ofile3 << numhr << endl;

    for (int i = 0; i < data.c.size(); i ++){
        ofile3 << i << " " << data.c[i].locationx << " " << data.c[i].locationy << " " << data.c[i].setupfee << std::endl;
    }

    for (int i = 0; i < data.vr.size(); i ++){
        ofile3 << data.vr[i].name << " " << data.vr[i].type << " " << data.vr[i].salary << endl;
    }
    for (int i = 0; i < data.nr.size(); i ++){
        ofile3 << data.nr[i].name << " " << data.nr[i].type << " " << data.nr[i].salary << endl;
    }
    for (int i = 0; i < data.hr.size(); i ++){
        ofile3 << data.hr[i].name << " " << data.hr[i].type << " " << data.hr[i].salary << endl;
    }


    for (int i = 0; i < data.p1.size(); i ++){
        if (data.p1[i].end + 1 < 12){
            data.p1[i].end = data.p1[i].end + 1;
        }
        else{
            data.p1[i].start = data.p1[i].start - 1;
        }
        ofile3 << data.p1[i].name << " " << data.p1[i].kind << " " << data.p1[i].locationx << " " << data.p1[i].locationy << " " << data.p1[i].servicetime << " " << data.p1[i].start << " " << data.p1[i].end << endl;
    }

    for (int i = 0; i < data.p2.size(); i ++){
        if (data.p2[i].end + 1 < 12){
            data.p2[i].end = data.p2[i].end + 1;
        }
        else{
            data.p2[i].start = data.p2[i].start - 1;
        }
        ofile3 << data.p2[i].name << " " << data.p2[i].kind << " " << data.p2[i].locationx << " " << data.p2[i].locationy << " " << data.p2[i].servicetime << " " << data.p2[i].start << " " << data.p2[i].end << endl;
    }
    ofile.close();
}

int main(){
    for (int i = 0; i < 10; i++) {
        generateData(numc, nump1, nump2, numvr, numnr, numhr, to_string(i));
    }

    return 0;
}
