"""
author: Antoine Spahr

date : 28.09.2020

----------

TO DO :

"""

def print_progessbar(N, Max, Name='', Size=10, end_char='', erase=False):
    """
    Print a progress bar. To be used in a for-loop and called at each iteration
    with the iteration number and the max number of iteration.
    ------------
    INPUT
        |---- N (int) the iteration current number
        |---- Max (int) the total number of iteration
        |---- Name (str) an optional name for the progress bar
        |---- Size (int) the size of the progress bar
        |---- end_char (str) the print end parameter to used in the end of the
        |                    progress bar (default is '')
        |---- erase (bool) whether to erase the progress bar when 100% is reached.
    OUTPUT
        |---- None
    """
    print(f'{Name} {N+1:04d}/{Max:04d}'.ljust(len(Name) + 12) \
        + f'|{"█"*int(Size*(N+1)/Max)}'.ljust(Size+1) + f'| {(N+1)/Max:.1%}'.ljust(6), \
        end='\r')

    if N+1 == Max:
        if erase:
            print(' '.ljust(len(Name) + Size + 40), end='\r')
        else:
            print('')
