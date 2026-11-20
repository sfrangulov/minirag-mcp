from minirag_mcp.__main__ import main

# Sync workers are spawned processes, and each runs this file again as `__mp_main__`:
# unguarded, every worker would start a second server on the client's stdin and stdout.
if __name__ == "__main__":
    main()
