#include "NurseVrp.hpp"


Label* Label::getParentLabel(){
    return parent;
}

void Label::linkToOther(Label *pre) {
    // cout << "in linktoother: " << pre->getEndVex() << endl;
    // cout << " Check cost in parent: " << pre->getCurCost() << endl;
    parent = pre;
}

int Label::getParent() {
    if (parent == nullptr) {
        return -1;
    }
    //#ifdef DEBUG_CHECK_PARENT
    //cout << " Check parent in getParent(): " << parent->getEndVex() << endl;
    //cout << " Check cost in parent: " << parent->getCurCost() << endl;
    //#endif
    return parent->getEndVex();
}













