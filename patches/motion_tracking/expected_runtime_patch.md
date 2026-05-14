# Expected Runtime Patch

The first runtime patch should be minimal and isolated to `motion_tracking`
runtime sources:

1. add a new source mode, e.g. `floodnet`
2. reuse the existing horizon consumption path
3. accept externally pushed frames or clips
4. preserve current `udp` and `vr` sources unchanged

The patch should not modify:

- policy architecture
- training code
- reward logic
- controller config semantics
