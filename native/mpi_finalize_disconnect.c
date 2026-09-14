/*
 * disconnect a dynamically spawned vic child from its parent before the
 * underlying mpi implementation finalizes the process.
 *
 * the established vic-mf6 launcher needed this interposer with openmpi 4.1.6
 * to keep repeated mpi_comm_spawn windows from hanging during shutdown. the
 * helper stays outside vic so the coupling-specific lifecycle workaround does
 * not alter unrelated vic execution.
 */

#include <mpi.h>

int MPI_Finalize(void)
{
    MPI_Comm parent = MPI_COMM_NULL;

    MPI_Comm_get_parent(&parent);
    if (parent != MPI_COMM_NULL) {
        MPI_Comm_disconnect(&parent);
    }

    return PMPI_Finalize();
}
