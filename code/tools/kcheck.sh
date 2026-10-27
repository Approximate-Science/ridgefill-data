#!/bin/bash
# kcheck.sh SINCE  -> prints amdgpu fault lines since SINCE; exit 1 if any.
# The brief's pattern, minus the lines a healthy card prints when it wakes from runtime suspend
# (since the 2026-10-01 reboot one card suspends when idle): "SMU is resuming/resumed successfully",
# "program CP_MES_CNTL", "MES: ... mask", and LACT's overdrive restore failing with ret:-5.
# Kept as faults: MES failed to respond, "SMU is in hanged state", ring timeouts, GPU reset,
# device lost / lost from bus, and the overdrive failure with ret:-121 (it came with the 09-30 SMU hang).
journalctl -k --since "$1" -o short-iso \
  | grep -E "amdgpu.*(MES|SMU|ring .* timeout|GPU reset|device lost|lost from bus|overdrive table, ret:-121)" \
  | grep -vE "SMU is resuming|SMU is resumed successfully|SMU is initialized successfully|program CP_MES_CNTL|MES: (vmid_mask|gfx_hqd_mask)"
[ ${PIPESTATUS[2]} -ne 0 ]
