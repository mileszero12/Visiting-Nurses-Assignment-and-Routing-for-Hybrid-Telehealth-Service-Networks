#include "NurseVrp.hpp"
#include <cstdlib>
#include <ctime>


void NurseVrp::LocalSearch() {
    int iter = 0;
    while (iter < LocalSearchNum) {
        bool flag = false;

        srand(time(0)); // generate a random number between 0 and 2
                        // 0 is insertion, 1 is deletion, 2 is relocation
        int randomNum = rand() % 3;

        // cout << "Random number: " << randomNum << endl;
        if (randomNum == 0) {
            cout << "Insertion" << endl;
            // Insertion();
        } else if (randomNum == 1) {
            cout << "Deletion" << endl;
            // Deletion();
        } else {
            cout << "Relocation" << endl;
            // Relocation();
        }

        if (flag){
            LocalSearchNum ++;
        }
    }

}
