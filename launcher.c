// Transcriber.app's executable: runs dictate.py with Python inside this
// process, so macOS treats it as Transcriber (its name, icon, Dock entry and
// permissions) rather than as uv or python.
//
// Built by build_app.sh with -DPROJECT="/path/to/transcriber".

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    (void)argc;
    if (chdir(PROJECT) != 0) {
        perror("chdir " PROJECT);
        return 1;
    }

    // Opened from Finder there's no terminal, so keep a log instead.
    if (!isatty(STDOUT_FILENO)) {
        char log[1024];
        snprintf(log, sizeof log, "%s/Library/Logs/Transcriber.log", getenv("HOME"));
        freopen(log, "a", stdout);
        freopen(log, "a", stderr);
    }

    PyConfig config;
    PyConfig_InitPythonConfig(&config);
    // Pointing at the project's venv python makes Python find its packages.
    PyConfig_SetBytesString(&config, &config.program_name, PROJECT "/.venv/bin/python");
    char *py_argv[] = {argv[0], PROJECT "/dictate.py"};
    PyStatus status = PyConfig_SetBytesArgv(&config, 2, py_argv);
    if (!PyStatus_Exception(status)) {
        status = Py_InitializeFromConfig(&config);
    }
    PyConfig_Clear(&config);
    if (PyStatus_Exception(status)) {
        Py_ExitStatusException(status);
    }
    return Py_RunMain();
}
