#include <iostream>
#include <string>

#include "replayer.h"

using namespace std;

int main(int argc, char** argv) {
    string filename;
    if (argc < 2) {
        cout << "Enter filename of recorded blob: ";
        cin >> filename;
    } else {
        filename = argv[1];
    }

    Replayer replayer;
    if (replayer.replay(filename) < 0) {
        cout << "Unable to replay '" << filename << "'\n";
    } else {
        cout << "'" << filename << "' was successfully replayed.'\n";
    }

    return 0;
}