#include <stdio.h>
#include "include/Attention.h"
#include "include/ClauseBank.h"
#include "include/ClauseBankSparse.h"
#include "include/AutoencoderClauses.h"
#include "include/AutoencoderDocuments.h"
#include "include/WeightBank.h"

#ifndef _WIN32
    // Linux/POSIX-specific headers
    #include <sys/stat.h>
    #include <sys/types.h>
#else
    // Windows-specific headers
    #include <direct.h>
#endif

int createDirectory(const char* path) {
#ifdef _WIN32
    // Windows-specific code
    if (_mkdir(path) == 0) {
        return 1;  // Directory successfully created
    } else {
        return 0;  // Failed to create the directory
    }
#else
    // Linux/POSIX-specific code
    if (mkdir(path, 0777) == 0) {
        return 1;  // Directory successfully created
    } else {
        return 0;  // Failed to create the directory
    }
#endif
}

int main() {
    const char* directory = "result";

    if (createDirectory(directory)) {
        printf("Directory created or already exists.\n");
    } else {
        printf("Failed to create the directory.\n");
    }

    printf("Hello, World!\n");
    return 0;
}
