0000000000016890 <_ZN21topsop_global_kernels21flash_attn_fwd_kernelIu6__bf16Li256EEEvPT_PfS2_S2_S2_S3_S2_Pi24VARLEN_ATTN_FWD_OP_PARAS>:
   16890: 00 00 00 00  	v.nop
   16894: 03 00 00 00 00 00    	<unknown>
   1689a: 00 00        	s.nop16
   1689c: 00 01        	s.nop16
   1689e: 11 20        	<unknown>
   168a0: 80 00 00 00  	v.nop
   168a4: 00 00 00 00 00 00    	m.nop
   168aa: 00 00        	s.nop16
   168ac: 00 00        	s.nop16
   168ae: 00 00        	<unknown>
   168b0: 00 00 00 00  	v.nop
   168b4: 00 84 16 00 18 01    	m.nop
   168ba: 00 00        	s.nop16
   168bc: 00 00        	s.nop16
   168be: 00 00        	<unknown>
   168c0: 01 00 00 00  	v.nop
   168c4: 04 00 00 00 04 00    	m.movsft_va.sft0 oacc0, oacc32, 0x0, oacc0, 0x0, r0
   168ca: 00 00        	s.nop16
   168cc: 60 0b        	s.or2 r13, r1
   168ce: 01 00        	c.nop
   168d0: 00 82 16 00  	v.nop
   168d4: 00 00 00 00 00 00    	m.nop
   168da: 00 00        	s.nop16
   168dc: 00 00        	s.nop16
   168de: 00 00        	<unknown>
   168e0: 00 00 00 00  	v.nop
   168e4: 00 00 00 00 00 00    	m.nop
   168ea: 00 00        	s.nop16
   168ec: 00 00        	s.nop16
   168ee: 00 00        	<unknown>
   168f0: 00 00 00 00  	v.nop
   168f4: 00 00 00 00 00 00    	m.nop
   168fa: 00 00        	s.nop16
   168fc: 00 00        	s.nop16
   168fe: 00 00        	<unknown>
   16900: 00 00 00 00  	v.nop
   16904: 00 00 00 00 00 00    	m.nop
   1690a: 00 00        	s.nop16
   1690c: 00 00        	s.nop16
   1690e: 00 00        	<unknown>
   16910: 00 00 00 00  	v.nop
   16914: 00 00 00 00 00 00    	m.nop
   1691a: a1 10 08 10  	l.subi.u sp, sp, 0x8
   1691e: 01 00        	c.nop
   16920: 00 00 00 00  	v.nop
   16924: 00 00 00 00 00 00    	m.nop
   1692a: 31 11 c3 00  	l.st.w lr, [sp, 0x4]
   1692e: 01 00        	c.nop
   16930: 00 00 00 00  	v.nop
   16934: 00 00 00 00 00 00    	m.nop
   1693a: 31 10 df 00  	l.st.w r31, [sp]
   1693e: 01 00        	c.nop
   16940: 00 00 00 00  	v.nop
   16944: 00 00 00 00 00 00    	m.nop
   1694a: e1 17 08 00  	l.addi.u r31, sp, 0x8
   1694e: 01 00        	c.nop
   16950: 00 00 00 00  	v.nop
   16954: 00 00 00 00 00 00    	m.nop
   1695a: 63 10 83 03  	l.srli r1, sp, 0x3
   1695e: 01 00        	c.nop
   16960: 00 00 00 00  	v.nop
   16964: 00 00 00 00 00 00    	m.nop
   1696a: a3 08 03 03  	l.slli sp, r1, 0x3
   1696e: 01 00        	c.nop
   16970: 00 00 00 00  	v.nop
   16974: 00 00 00 00 00 00    	m.nop
   1697a: a1 10 c0 1d  	l.subi.u sp, sp, 0xdc0
   1697e: 01 00        	c.nop
   16980: 00 00 00 00  	v.nop
   16984: 00 00 00 00 00 00    	m.nop
   1698a: 2f 27 6d 21  	l.ldi16.s r4, 0xdbc
   1698e: 01 44        	<unknown>
   16990: 00 00 00 00  	v.nop
   16994: 00 00 00 00 00 00    	m.nop
   1699a: 23 11 04 18  	<unknown>
   1699e: 01 00        	c.nop
   169a0: 00 00 00 00  	v.nop
   169a4: 00 00 00 00 00 00    	m.nop
   169aa: 31 20 c3 00  	l.st.w lr, [r4]
   169ae: 01 00        	c.nop
   169b0: 00 00 00 00  	v.nop
   169b4: 00 00 00 00 00 00    	m.nop
   169ba: 31 f9 44 00  	l.ld.w r4, [r31, 0x4]
   169be: 01 00        	c.nop
   169c0: 00 00 00 00  	v.nop
   169c4: 00 00 00 00 00 00    	m.nop
   169ca: 2f 33 03 21  	l.ldi16.s r6, 0x6c
   169ce: 01 00        	c.nop
   169d0: 00 00 00 00  	v.nop
   169d4: 00 00 00 00 00 00    	m.nop
   169da: f1 fe 4c 00  	l.ld.w r27, [r31, 0xc]
   169de: 01 00        	c.nop
   169e0: 00 00 00 00  	v.nop
   169e4: 00 00 00 00 00 00    	m.nop
   169ea: b1 fb 54 00  	l.ld.w r14, [r31, 0x14]
   169ee: 01 00        	c.nop
   169f0: 00 00 00 00  	v.nop
   169f4: 00 00 00 00 00 00    	m.nop
   169fa: ef e7 07 21  	l.ldi16.s r28, 0xff
   169fe: 01 00        	c.nop
   16a00: 00 00 00 00  	v.nop
   16a04: 00 00 00 00 00 00    	m.nop
   16a0a: 31 10 c4 1a  	l.st.w r4, [sp, 0x1a0]
   16a0e: 01 00        	c.nop
   16a10: 00 00 00 00  	v.nop
   16a14: 00 00 00 00 00 00    	m.nop
   16a1a: 31 f9 40 00  	l.ld.w r4, [r31]
   16a1e: 01 00        	c.nop
   16a20: 00 00 00 00  	v.nop
   16a24: 00 00 00 00 00 00    	m.nop
   16a2a: 31 17 c4 18  	l.st.w r4, [sp, 0x19c]
   16a2e: 01 00        	c.nop
   16a30: 00 00 00 00  	v.nop
   16a34: 00 00 00 00 00 00    	m.nop
   16a3a: 31 f9 48 00  	l.ld.w r4, [r31, 0x8]
   16a3e: 01 00        	c.nop
   16a40: 00 00 00 00  	v.nop
   16a44: 00 00 00 00 00 00    	m.nop
   16a4a: 31 14 c4 1c  	l.st.w r4, [sp, 0x1d0]
   16a4e: 01 00        	c.nop
   16a50: 00 00 00 00  	v.nop
   16a54: 00 00 00 00 00 00    	m.nop
   16a5a: 31 f9 50 00  	l.ld.w r4, [r31, 0x10]
   16a5e: 01 00        	c.nop
   16a60: 00 00 00 00  	v.nop
   16a64: 00 00 00 00 00 00    	m.nop
   16a6a: 31 16 c4 0e  	l.st.w r4, [sp, 0xf8]
   16a6e: 01 00        	c.nop
   16a70: 00 00 00 00  	v.nop
   16a74: 00 00 00 00 00 00    	m.nop
   16a7a: 31 f9 5c 00  	l.ld.w r4, [r31, 0x1c]
   16a7e: 01 00        	c.nop
   16a80: 00 00 00 00  	v.nop
   16a84: 00 00 00 00 00 00    	m.nop
   16a8a: 31 15 c4 04  	l.st.w r4, [sp, 0x54]
   16a8e: 01 00        	c.nop
   16a90: 00 00 00 00  	v.nop
   16a94: 00 00 00 00 00 00    	m.nop
   16a9a: 31 f9 58 00  	l.ld.w r4, [r31, 0x18]
   16a9e: 01 00        	c.nop
   16aa0: 00 00 00 00  	v.nop
   16aa4: 00 00 00 00 00 00    	m.nop
   16aaa: 31 17 c4 04  	l.st.w r4, [sp, 0x5c]
   16aae: 01 00        	c.nop
   16ab0: 00 00 00 00  	v.nop
   16ab4: 00 00 00 00 00 00    	m.nop
   16aba: 31 f9 44 02  	l.ld.w r4, [r31, 0x24]
   16abe: 01 00        	c.nop
   16ac0: 00 00 00 00  	v.nop
   16ac4: 00 00 00 00 00 00    	m.nop
   16aca: 31 14 c4 04  	l.st.w r4, [sp, 0x50]
   16ace: 01 00        	c.nop
   16ad0: 00 00 00 00  	v.nop
   16ad4: 00 00 00 00 00 00    	m.nop
   16ada: 31 f9 40 02  	l.ld.w r4, [r31, 0x20]
   16ade: 01 00        	c.nop
   16ae0: 00 00 00 00  	v.nop
   16ae4: 00 00 00 00 00 00    	m.nop
   16aea: 31 16 c4 04  	l.st.w r4, [sp, 0x58]
   16aee: 01 00        	c.nop
   16af0: 00 00 00 00  	v.nop
   16af4: 00 00 00 00 00 00    	m.nop
   16afa: 31 f9 4c 02  	l.ld.w r4, [r31, 0x2c]
   16afe: 01 00        	c.nop
   16b00: 00 00 00 00  	v.nop
   16b04: 00 00 00 00 00 00    	m.nop
   16b0a: 31 13 c4 18  	l.st.w r4, [sp, 0x18c]
   16b0e: 01 00        	c.nop
   16b10: 00 00 00 00  	v.nop
   16b14: 00 00 00 00 00 00    	m.nop
   16b1a: 31 f9 48 02  	l.ld.w r4, [r31, 0x28]
   16b1e: 01 00        	c.nop
   16b20: 00 00 00 00  	v.nop
   16b24: 00 00 00 00 00 00    	m.nop
   16b2a: 31 12 c4 18  	l.st.w r4, [sp, 0x188]
   16b2e: 01 00        	c.nop
   16b30: 00 00 00 00  	v.nop
   16b34: 00 00 00 00 00 00    	m.nop
   16b3a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   16b3e: 01 00        	c.nop
   16b40: 00 00 00 00  	v.nop
   16b44: 00 00 00 00 00 00    	m.nop
   16b4a: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   16b4e: 01 00        	c.nop
   16b50: 00 00 00 00  	v.nop
   16b54: 00 00 00 00 00 00    	m.nop
   16b5a: f3 74 80 00  	l.movs r19, mbox_extra_sts
   16b5e: 01 00        	c.nop
   16b60: 00 00 00 00  	v.nop
   16b64: 00 00 00 00 00 00    	m.nop
   16b6a: a3 21 06 10  	<unknown>
   16b6e: 01 00        	c.nop
   16b70: 00 00 00 00  	v.nop
   16b74: 00 00 00 00 00 00    	m.nop
   16b7a: 71 35 40 00  	l.ld.w r21, [r6]
   16b7e: 01 00        	c.nop
   16b80: 00 00 00 00  	v.nop
   16b84: 00 00 00 00 00 00    	m.nop
   16b8a: b1 35 44 00  	l.ld.w r22, [r6, 0x4]
   16b8e: 01 00        	c.nop
   16b90: 00 00 00 00  	v.nop
   16b94: 00 00 00 00 00 00    	m.nop
   16b9a: 31 34 4c 00  	l.ld.w r16, [r6, 0xc]
   16b9e: 01 00        	c.nop
   16ba0: 00 00 00 00  	v.nop
   16ba4: 00 00 00 00 00 00    	m.nop
   16baa: 31 35 50 00  	l.ld.w r20, [r6, 0x10]
   16bae: 01 00        	c.nop
   16bb0: 00 00 00 00  	v.nop
   16bb4: 00 00 00 00 00 00    	m.nop
   16bba: b1 34 54 00  	l.ld.w r18, [r6, 0x14]
   16bbe: 01 00        	c.nop
   16bc0: 00 00 00 00  	v.nop
   16bc4: 00 00 00 00 00 00    	m.nop
   16bca: 73 7c 80 00  	<unknown>
   16bce: 01 00        	c.nop
   16bd0: 00 00 00 00  	v.nop
   16bd4: 00 00 00 00 00 00    	m.nop
   16bda: 71 f9 50 08  	l.ld.w r5, [r31, 0x90]
   16bde: 01 00        	c.nop
   16be0: 00 00 00 00  	v.nop
   16be4: 00 00 00 00 00 00    	m.nop
   16bea: f1 f9 13 10  	l.ld.b r7, [r31, 0x113]
   16bee: 01 00        	c.nop
   16bf0: 00 00 00 00  	v.nop
   16bf4: 00 00 00 00 00 00    	m.nop
   16bfa: f1 33 48 00  	l.ld.w r15, [r6, 0x8]
   16bfe: 01 00        	c.nop
   16c00: 00 00 00 00  	v.nop
   16c04: 00 00 00 00 00 00    	m.nop
   16c0a: b1 f9 44 08  	l.ld.w r6, [r31, 0x84]
   16c0e: 01 00        	c.nop
   16c10: 00 00 00 00  	v.nop
   16c14: 00 00 00 00 00 00    	m.nop
   16c1a: 31 fe 4c 08  	l.ld.w r24, [r31, 0x8c]
   16c1e: 01 00        	c.nop
   16c20: 00 00 00 00  	v.nop
   16c24: 00 00 00 00 00 00    	m.nop
   16c2a: 31 14 c5 08  	l.st.w r5, [sp, 0x90]
   16c2e: 01 00        	c.nop
   16c30: 00 00 00 00  	v.nop
   16c34: 00 00 00 00 00 00    	m.nop
   16c3a: 71 f9 48 06  	l.ld.w r5, [r31, 0x68]
   16c3e: 01 00        	c.nop
   16c40: 00 00 00 00  	v.nop
   16c44: 00 00 00 00 00 00    	m.nop
   16c4a: f1 fd 54 06  	l.ld.w r23, [r31, 0x74]
   16c4e: 01 00        	c.nop
   16c50: 00 00 00 00  	v.nop
   16c54: 00 00 00 00 00 00    	m.nop
   16c5a: 71 fa 0f 10  	l.ld.b r9, [r31, 0x10f]
   16c5e: 01 00        	c.nop
   16c60: 00 00 00 00  	v.nop
   16c64: 00 00 00 00 00 00    	m.nop
   16c6a: e5 39 9c 01  	l.andix r7, r7, 0x1c
   16c6e: 01 00        	c.nop
   16c70: 00 00 00 00  	v.nop
   16c74: 00 00 00 00 00 00    	m.nop
   16c7a: 31 17 c5 02  	l.st.w r5, [sp, 0x3c]
   16c7e: 01 00        	c.nop
   16c80: 00 00 00 00  	v.nop
   16c84: 00 00 00 00 00 00    	m.nop
   16c8a: 71 f9 40 10  	l.ld.w r5, [r31, 0x100]
   16c8e: 01 00        	c.nop
   16c90: 00 00 00 00  	v.nop
   16c94: 00 00 00 00 00 00    	m.nop
   16c9a: 31 16 c5 22  	l.st.w r5, [sp, 0x238]
   16c9e: 01 00        	c.nop
   16ca0: 00 00 00 00  	v.nop
   16ca4: 00 00 00 00 00 00    	m.nop
   16caa: 71 f9 4c 06  	l.ld.w r5, [r31, 0x6c]
   16cae: 01 00        	c.nop
   16cb0: 00 00 00 00  	v.nop
   16cb4: 00 00 00 00 00 00    	m.nop
   16cba: 31 15 c5 02  	l.st.w r5, [sp, 0x34]
   16cbe: 01 00        	c.nop
   16cc0: 00 00 00 00  	v.nop
   16cc4: 00 00 00 00 00 00    	m.nop
   16cca: 71 f9 5c 08  	l.ld.w r5, [r31, 0x9c]
   16cce: 01 00        	c.nop
   16cd0: 00 00 00 00  	v.nop
   16cd4: 00 00 00 00 00 00    	m.nop
   16cda: 31 11 c5 18  	l.st.w r5, [sp, 0x184]
   16cde: 01 00        	c.nop
   16ce0: 00 00 00 00  	v.nop
   16ce4: 00 00 00 00 00 00    	m.nop
   16cea: 71 f9 44 10  	l.ld.w r5, [r31, 0x104]
   16cee: 01 00        	c.nop
   16cf0: 00 00 00 00  	v.nop
   16cf4: 00 00 00 00 00 00    	m.nop
   16cfa: 31 11 c5 12  	l.st.w r5, [sp, 0x124]
   16cfe: 01 00        	c.nop
   16d00: 00 00 00 00  	v.nop
   16d04: 00 00 00 00 00 00    	m.nop
   16d0a: fb 39 00 40  	<unknown>
   16d0e: 01 00        	c.nop
   16d10: 00 00 00 00  	v.nop
   16d14: 00 00 00 00 00 00    	m.nop
   16d1a: 31 15 c8 08  	l.st.w r8, [sp, 0x94]
   16d1e: 01 00        	c.nop
   16d20: 00 00 00 00  	v.nop
   16d24: 00 00 00 00 00 00    	m.nop
   16d2a: a5 4e 9c 01  	l.andix r26, r9, 0x1c
   16d2e: 01 00        	c.nop
   16d30: 00 00 00 00  	v.nop
   16d34: 00 00 00 00 00 00    	m.nop
   16d3a: 3b 3a 08 00  	<unknown>
   16d3e: 01 00        	c.nop
   16d40: 00 00 00 00  	v.nop
   16d44: 00 00 00 00 00 00    	m.nop
   16d4a: 00 00        	s.nop16
   16d4c: 00 00        	s.nop16
   16d4e: 85 44        	<unknown>
   16d50: 00 00 00 00  	v.nop
   16d54: 00 00 00 00 00 00    	m.nop
   16d5a: 3b d2 08 00  	<unknown>
   16d5e: 01 00        	c.nop
   16d60: 00 00 00 00  	v.nop
   16d64: 00 00 00 00 00 00    	m.nop
   16d6a: 2b 40 80 01  	l.jal r0, 0x8 <0x16de0>
   16d6e: 01 00        	c.nop
   16d70: 00 00 00 00  	v.nop
   16d74: 00 00 00 00 00 00    	m.nop
   16d7a: 00 00        	s.nop16
   16d7c: 00 00        	s.nop16
   16d7e: 89 42        	<unknown>
   16d80: 00 00 00 00  	v.nop
   16d84: 00 00 00 00 00 00    	m.nop
   16d8a: 31 15 c5 08  	l.st.w r5, [sp, 0x94]
   16d8e: 01 00        	c.nop
   16d90: 00 00 00 00  	v.nop
   16d94: 00 00 00 00 00 00    	m.nop
   16d9a: a5 4e 9c 01  	l.andix r26, r9, 0x1c
   16d9e: 01 00        	c.nop
   16da0: 00 00 00 00  	v.nop
   16da4: 00 00 00 00 00 00    	m.nop
   16daa: bb 3e 68 7f  	<unknown>
   16dae: 01 00        	c.nop
   16db0: 00 00 00 00  	v.nop
   16db4: 00 00 00 00 00 00    	m.nop
   16dba: 00 00        	s.nop16
   16dbc: 00 00        	s.nop16
   16dbe: fd 54        	<unknown>
   16dc0: 00 00 00 00  	v.nop
   16dc4: 00 00 00 00 00 00    	m.nop
   16dca: bb d0 08 40  	<unknown>
   16dce: 01 00        	c.nop
   16dd0: 00 00 00 00  	v.nop
   16dd4: 00 00 00 00 00 00    	m.nop
   16dda: 31 15 c9 08  	l.st.w r9, [sp, 0x94]
   16dde: 01 00        	c.nop
   16de0: 00 00 00 00  	v.nop
   16de4: 00 00 00 00 00 00    	m.nop
   16dea: 71 11 5c 02  	l.ld.w r5, [sp, 0x3c]
   16dee: 01 00        	c.nop
   16df0: 00 00 00 00  	v.nop
   16df4: 00 00 00 00 00 00    	m.nop
   16dfa: 6f 40 60 1b  	l.ldi16.u r8, 0xdc01
   16dfe: 01 00        	c.nop
   16e00: 00 00 00 00  	v.nop
   16e04: 00 00 00 00 00 00    	m.nop
   16e0a: 6f 42 80 00  	l.ldi_hi r8, 0x9
   16e0e: 01 00        	c.nop
   16e10: 00 00 00 00  	v.nop
   16e14: 00 00 00 00 00 00    	m.nop
   16e1a: bb 28 88 c0  	l.blt.s r5, r8, 0x2 <0x16e30>
   16e1e: 01 00        	c.nop
   16e20: 00 00 00 00  	v.nop
   16e24: 00 00 00 00 00 00    	m.nop
   16e2a: 2b 08 98 05  	l.jal r0, 0x1181 <0x28630>
   16e2e: 01 00        	c.nop
   16e30: 00 00 00 00  	v.nop
   16e34: 00 00 00 00 00 00    	m.nop
   16e3a: 71 11 5c 04  	l.ld.w r5, [sp, 0x5c]
   16e3e: 81 44        	<unknown>
   16e40: 00 00 00 00  	v.nop
   16e44: 00 00 00 00 00 00    	m.nop
   16e4a: 31 12 54 04  	l.ld.w r8, [sp, 0x54]
   16e4e: 01 00        	c.nop
   16e50: 00 00 00 00  	v.nop
   16e54: 00 00 00 00 00 00    	m.nop
   16e5a: 31 11 c7 04  	l.st.w r7, [sp, 0x44]
   16e5e: 01 00        	c.nop
   16e60: 00 00 00 00  	v.nop
   16e64: 00 00 00 00 00 00    	m.nop
   16e6a: ef 08 00 21  	l.ldi16.s r1, 0x3
   16e6e: 01 00        	c.nop
   16e70: 00 00 00 00  	v.nop
   16e74: 00 00 00 00 00 00    	m.nop
   16e7a: 27 28 09 01  	<unknown>
   16e7e: 01 00        	c.nop
   16e80: 00 00 00 00  	v.nop
   16e84: 00 00 00 00 00 00    	m.nop
   16e8a: 61 08 01 70  	l.subi.s r1, r1, 0x1
   16e8e: 01 00        	c.nop
   16e90: 00 00 00 00  	v.nop
   16e94: 00 00 00 00 00 00    	m.nop
   16e9a: 27 40 09 01  	<unknown>
   16e9e: 01 00        	c.nop
   16ea0: 00 00 00 00  	v.nop
   16ea4: 00 00 00 00 00 00    	m.nop
   16eaa: 61 08 01 70  	l.subi.s r1, r1, 0x1
   16eae: 01 00        	c.nop
   16eb0: 00 00 00 00  	v.nop
   16eb4: 00 00 00 00 00 00    	m.nop
   16eba: 23 0a 81 03  	l.srli r8, r1, 0x1
   16ebe: 01 00        	c.nop
   16ec0: 00 00 00 00  	v.nop
   16ec4: 00 00 00 00 00 00    	m.nop
   16eca: 31 16 ce 06  	l.st.w r14, [sp, 0x78]
   16ece: 01 00        	c.nop
   16ed0: 00 00 00 00  	v.nop
   16ed4: 00 00 00 00 00 00    	m.nop
   16eda: bb 44 00 00  	<unknown>
   16ede: 01 00        	c.nop
   16ee0: 00 00 00 00  	v.nop
   16ee4: 00 00 00 00 00 00    	m.nop
   16eea: 71 11 58 04  	l.ld.w r5, [sp, 0x58]
   16eee: 85 43        	<unknown>
   16ef0: 00 00 00 00  	v.nop
   16ef4: 00 00 00 00 00 00    	m.nop
   16efa: 31 12 50 04  	l.ld.w r8, [sp, 0x50]
   16efe: 01 00        	c.nop
   16f00: 00 00 00 00  	v.nop
   16f04: 00 00 00 00 00 00    	m.nop
   16f0a: ef 08 00 21  	l.ldi16.s r1, 0x3
   16f0e: 01 00        	c.nop
   16f10: 00 00 00 00  	v.nop
   16f14: 00 00 00 00 00 00    	m.nop
   16f1a: 27 28 09 01  	<unknown>
   16f1e: 01 00        	c.nop
   16f20: 00 00 00 00  	v.nop
   16f24: 00 00 00 00 00 00    	m.nop
   16f2a: 61 08 01 70  	l.subi.s r1, r1, 0x1
   16f2e: 01 00        	c.nop
   16f30: 00 00 00 00  	v.nop
   16f34: 00 00 00 00 00 00    	m.nop
   16f3a: 27 40 09 01  	<unknown>
   16f3e: 01 00        	c.nop
   16f40: 00 00 00 00  	v.nop
   16f44: 00 00 00 00 00 00    	m.nop
   16f4a: 61 08 01 70  	l.subi.s r1, r1, 0x1
   16f4e: 01 00        	c.nop
   16f50: 00 00 00 00  	v.nop
   16f54: 00 00 00 00 00 00    	m.nop
   16f5a: 63 0a 81 03  	l.srli r9, r1, 0x1
   16f5e: 01 00        	c.nop
   16f60: 00 00 00 00  	v.nop
   16f64: 00 00 00 00 00 00    	m.nop
   16f6a: 63 c1 06 19  	<unknown>
   16f6e: 01 00        	c.nop
   16f70: 00 00 00 00  	v.nop
   16f74: 00 00 00 00 00 00    	m.nop
   16f7a: 31 15 c5 06  	l.st.w r5, [sp, 0x74]
   16f7e: 01 00        	c.nop
   16f80: 00 00 00 00  	v.nop
   16f84: 00 00 00 00 00 00    	m.nop
   16f8a: 71 11 50 08  	l.ld.w r5, [sp, 0x90]
   16f8e: 01 00        	c.nop
   16f90: 00 00 00 00  	v.nop
   16f94: 00 00 00 00 00 00    	m.nop
   16f9a: 63 29 06 19  	<unknown>
   16f9e: 01 00        	c.nop
   16fa0: 00 00 00 00  	v.nop
   16fa4: 00 00 00 00 00 00    	m.nop
   16faa: 31 10 c5 02  	l.st.w r5, [sp, 0x20]
   16fae: 01 00        	c.nop
   16fb0: 00 00 00 00  	v.nop
   16fb4: 00 00 00 00 00 00    	m.nop
   16fba: bb 4f 00 00  	<unknown>
   16fbe: 01 00        	c.nop
   16fc0: 00 00 00 00  	v.nop
   16fc4: 00 00 00 00 00 00    	m.nop
   16fca: 31 12 c7 02  	l.st.w r7, [sp, 0x28]
   16fce: 01 00        	c.nop
   16fd0: 00 00 00 00  	v.nop
   16fd4: 00 00 00 00 00 00    	m.nop
   16fda: 31 14 d7 02  	l.st.w r23, [sp, 0x30]
   16fde: 01 00        	c.nop
   16fe0: 00 00 00 00  	v.nop
   16fe4: 00 00 00 00 00 00    	m.nop
   16fea: 2b 58 82 01  	l.jal r0, 0x2b <0x17290>
   16fee: 01 00        	c.nop
   16ff0: 00 00 00 00  	v.nop
   16ff4: 00 00 00 00 00 00    	m.nop
   16ffa: 71 11 5c 02  	l.ld.w r5, [sp, 0x3c]
   16ffe: 8d 43        	<unknown>
   17000: 00 00 00 00  	v.nop
   17004: 00 00 00 00 00 00    	m.nop
   1700a: 31 12 54 02  	l.ld.w r8, [sp, 0x34]
   1700e: 01 00        	c.nop
   17010: 00 00 00 00  	v.nop
   17014: 00 00 00 00 00 00    	m.nop
   1701a: 31 12 c7 02  	l.st.w r7, [sp, 0x28]
   1701e: 01 00        	c.nop
   17020: 00 00 00 00  	v.nop
   17024: 00 00 00 00 00 00    	m.nop
   1702a: 31 14 c5 02  	l.st.w r5, [sp, 0x30]
   1702e: 01 00        	c.nop
   17030: 00 00 00 00  	v.nop
   17034: 00 00 00 00 00 00    	m.nop
   1703a: 63 42 05 19  	<unknown>
   1703e: 01 00        	c.nop
   17040: 00 00 00 00  	v.nop
   17044: 00 00 00 00 00 00    	m.nop
   1704a: 63 c1 06 19  	<unknown>
   1704e: 01 00        	c.nop
   17050: 00 00 00 00  	v.nop
   17054: 00 00 00 00 00 00    	m.nop
   1705a: e3 4a 9f 0b  	l.srai r11, r9, 0x1f
   1705e: 01 00        	c.nop
   17060: 00 00 00 00  	v.nop
   17064: 00 00 00 00 00 00    	m.nop
   1706a: 31 10 c5 02  	l.st.w r5, [sp, 0x20]
   1706e: 86 05        	<unknown>
   17070: 00 00 00 00  	v.nop
   17074: 00 00 00 00 00 00    	m.nop
   1707a: 31 15 c5 06  	l.st.w r5, [sp, 0x74]
   1707e: 01 00        	c.nop
   17080: 00 00 00 00  	v.nop
   17084: 00 00 00 00 00 00    	m.nop
   1708a: 71 11 58 0e  	l.ld.w r5, [sp, 0xf8]
   1708e: 01 00        	c.nop
   17090: 00 00 00 00  	v.nop
   17094: 00 00 00 00 00 00    	m.nop
   1709a: a3 4a 01 03  	l.slli r10, r9, 0x1
   1709e: 01 00        	c.nop
   170a0: 00 00 00 00  	v.nop
   170a4: 00 00 00 00 00 00    	m.nop
   170aa: 23 53 9f 0b  	l.srai r12, r10, 0x1f
   170ae: 01 00        	c.nop
   170b0: 00 00 00 00  	v.nop
   170b4: 00 00 00 00 00 00    	m.nop
   170ba: 63 4b 9f 03  	l.srli r13, r9, 0x1f
   170be: 06 06        	<unknown>
   170c0: 00 00 00 00  	v.nop
   170c4: 00 00 00 00 00 00    	m.nop
   170ca: 63 53 9f 03  	l.srli r13, r10, 0x1f
   170ce: d5 8d        	<unknown>
   170d0: 00 00 00 00  	v.nop
   170d4: 00 00 00 00 00 00    	m.nop
   170da: 23 2a 0a 10  	<unknown>
   170de: 8a 04        	<unknown>
   170e0: 00 00 00 00  	v.nop
   170e4: 00 00 00 00 00 00    	m.nop
   170ea: a5 62 0d 02  	l.orix r10, r12, 0xd
   170ee: 01 00        	c.nop
   170f0: 00 00 00 00  	v.nop
   170f4: 00 00 00 00 00 00    	m.nop
   170fa: 31 17 c8 04  	l.st.w r8, [sp, 0x5c]
   170fe: 01 00        	c.nop
   17100: 00 00 00 00  	v.nop
   17104: 00 00 00 00 00 00    	m.nop
   1710a: 27 40 05 00  	l.slt.u r8, r5
   1710e: 01 00        	c.nop
   17110: 00 00 00 00  	v.nop
   17114: 00 00 00 00 00 00    	m.nop
   1711a: 23 72 0b 10  	<unknown>
   1711e: 01 00        	c.nop
   17120: 00 00 00 00  	v.nop
   17124: 00 00 00 00 00 00    	m.nop
   1712a: 21 42 01 40  	l.addi.u r8, r8, 0x1
   1712e: 01 00        	c.nop
   17130: 00 00 00 00  	v.nop
   17134: 00 00 00 00 00 00    	m.nop
   1713a: 31 15 c8 04  	l.st.w r8, [sp, 0x54]
   1713e: 01 00        	c.nop
   17140: 00 00 00 00  	v.nop
   17144: 00 00 00 00 00 00    	m.nop
   1714a: 23 2a 09 10  	<unknown>
   1714e: 01 00        	c.nop
   17150: 00 00 00 00  	v.nop
   17154: 00 00 00 00 00 00    	m.nop
   1715a: 31 16 c8 04  	l.st.w r8, [sp, 0x58]
   1715e: 01 00        	c.nop
   17160: 00 00 00 00  	v.nop
   17164: 00 00 00 00 00 00    	m.nop
   1716a: 27 40 05 00  	l.slt.u r8, r5
   1716e: 01 00        	c.nop
   17170: 00 00 00 00  	v.nop
   17174: 00 00 00 00 00 00    	m.nop
   1717a: 63 71 0a 10  	<unknown>
   1717e: 01 00        	c.nop
   17180: 00 00 00 00  	v.nop
   17184: 00 00 00 00 00 00    	m.nop
   1718a: 2b 78 80 01  	l.jal r0, 0xf <0x17270>
   1718e: 01 00        	c.nop
   17190: 00 00 00 00  	v.nop
   17194: 00 00 00 00 00 00    	m.nop
   1719a: 71 11 54 02  	l.ld.w r5, [sp, 0x34]
   1719e: 01 00        	c.nop
   171a0: 00 00 00 00  	v.nop
   171a4: 00 00 00 00 00 00    	m.nop
   171aa: 31 12 54 04  	l.ld.w r8, [sp, 0x54]
   171ae: 01 00        	c.nop
   171b0: 00 00 00 00  	v.nop
   171b4: 00 00 00 00 00 00    	m.nop
   171ba: 31 14 d7 02  	l.st.w r23, [sp, 0x30]
   171be: 01 00        	c.nop
   171c0: 00 00 00 00  	v.nop
   171c4: 00 00 00 00 00 00    	m.nop
   171ca: 63 2a 17 19  	<unknown>
   171ce: 89 42        	<unknown>
   171d0: 00 00 00 00  	v.nop
   171d4: 00 00 00 00 00 00    	m.nop
   171da: 31 12 c5 02  	l.st.w r5, [sp, 0x28]
   171de: 01 00        	c.nop
   171e0: 00 00 00 00  	v.nop
   171e4: 00 00 00 00 00 00    	m.nop
   171ea: 71 11 5c 04  	l.ld.w r5, [sp, 0x5c]
   171ee: 01 00        	c.nop
   171f0: 00 00 00 00  	v.nop
   171f4: 00 00 00 00 00 00    	m.nop
   171fa: a3 4a 9f 0b  	l.srai r10, r9, 0x1f
   171fe: 01 00        	c.nop
   17200: 00 00 00 00  	v.nop
   17204: 00 00 00 00 00 00    	m.nop
   1720a: e3 4a 9f 03  	l.srli r11, r9, 0x1f
   1720e: 86 04        	<unknown>
   17210: 00 00 00 00  	v.nop
   17214: 00 00 00 00 00 00    	m.nop
   1721a: 00 00        	s.nop16
   1721c: 00 00        	s.nop16
   1721e: 06 05        	<unknown>
   17220: 00 00 00 00  	v.nop
   17224: 00 00 00 00 00 00    	m.nop
   1722a: 00 00        	s.nop16
   1722c: 00 00        	s.nop16
   1722e: 4d 8d        	<unknown>
   17230: 00 00 00 00  	v.nop
   17234: 00 00 00 00 00 00    	m.nop
   1723a: 63 2a 09 10  	<unknown>
   1723e: 01 00        	c.nop
   17240: 00 00 00 00  	v.nop
   17244: 00 00 00 00 00 00    	m.nop
   1724a: 31 16 c9 04  	l.st.w r9, [sp, 0x58]
   1724e: 01 00        	c.nop
   17250: 00 00 00 00  	v.nop
   17254: 00 00 00 00 00 00    	m.nop
   1725a: 27 48 05 00  	l.slt.u r9, r5
   1725e: 01 00        	c.nop
   17260: 00 00 00 00  	v.nop
   17264: 00 00 00 00 00 00    	m.nop
   1726a: 63 41 0a 10  	<unknown>
   1726e: 01 00        	c.nop
   17270: 00 00 00 00  	v.nop
   17274: 00 00 00 00 00 00    	m.nop
   1727a: 61 29 01 40  	l.addi.u r5, r5, 0x1
   1727e: 01 00        	c.nop
   17280: 00 00 00 00  	v.nop
   17284: 00 00 00 00 00 00    	m.nop
   1728a: 31 14 c5 04  	l.st.w r5, [sp, 0x50]
   1728e: 01 00        	c.nop
   17290: 00 00 00 00  	v.nop
   17294: 00 00 00 00 00 00    	m.nop
   1729a: 71 22 40 00  	l.ld.w r9, [r4]
   1729e: 7d 57        	<unknown>
   172a0: 00 00 00 00  	v.nop
   172a4: 00 00 00 00 00 00    	m.nop
   172aa: 2f 40 01 21  	l.ldi16.s r8, 0x20
   172ae: 01 00        	c.nop
   172b0: 00 00 00 00  	v.nop
   172b4: 00 00 00 00 00 00    	m.nop
   172ba: a5 4a 8e 02  	l.xorix r10, r9, 0xe
   172be: 01 00        	c.nop
   172c0: 00 00 00 00  	v.nop
   172c4: 00 00 00 00 00 00    	m.nop
   172ca: 29 53 00 12  	l.bc.lz.u32 r12, r10
   172ce: 01 00        	c.nop
   172d0: 00 00 00 00  	v.nop
   172d4: 00 00 00 00 00 00    	m.nop
   172da: bb 60 88 c0  	l.blt.s r12, r8, 0x2 <0x172f0>
   172de: 01 00        	c.nop
   172e0: 00 00 00 00  	v.nop
   172e4: 00 00 00 00 00 00    	m.nop
   172ea: 2b 00 95 05  	l.jal r0, 0x1150 <0x287e0>
   172ee: 01 00        	c.nop
   172f0: 00 00 00 00  	v.nop
   172f4: 00 00 00 00 00 00    	m.nop
   172fa: 2f 40 80 30  	l.ldi_hi.m1 r8, 0x8000
   172fe: 91 42        	<unknown>
   17300: 00 00 00 00  	v.nop
   17304: 00 00 00 00 00 00    	m.nop
   1730a: a3 42 8c 02  	l.srl     r10, r8, r12
   1730e: fd 46        	<unknown>
   17310: 00 00 00 00  	v.nop
   17314: 00 00 00 00 00 00    	m.nop
   1731a: 65 52 09 02  	l.orix r9, r10, 0x9
   1731e: 01 00        	c.nop
   17320: 00 00 00 00  	v.nop
   17324: 00 00 00 00 00 00    	m.nop
   1732a: e3 22 05 10  	<unknown>
   1732e: a1 42        	<unknown>
   17330: 00 00 00 00  	v.nop
   17334: 00 00 00 00 00 00    	m.nop
   1733a: 31 20 c9 00  	l.st.w r9, [r4]
   1733e: 01 00        	c.nop
   17340: 00 00 00 00  	v.nop
   17344: 00 00 00 00 00 00    	m.nop
   1734a: 71 5a 40 00  	l.ld.w r9, [r11]
   1734e: 01 00        	c.nop
   17350: 00 00 00 00  	v.nop
   17354: 00 00 00 00 00 00    	m.nop
   1735a: 31 15 ca 16  	l.st.w r10, [sp, 0x174]
   1735e: 01 00        	c.nop
   17360: 00 00 00 00  	v.nop
   17364: 00 00 00 00 00 00    	m.nop
   1736a: 31 13 cb 04  	l.st.w r11, [sp, 0x4c]
   1736e: 01 00        	c.nop
   17370: 00 00 00 00  	v.nop
   17374: 00 00 00 00 00 00    	m.nop
   1737a: 31 10 da 12  	l.st.w r26, [sp, 0x120]
   1737e: 01 00        	c.nop
   17380: 00 00 00 00  	v.nop
   17384: 00 00 00 00 00 00    	m.nop
   1738a: 31 13 c7 1c  	l.st.w r7, [sp, 0x1cc]
   1738e: 01 00        	c.nop
   17390: 00 00 00 00  	v.nop
   17394: 00 00 00 00 00 00    	m.nop
   1739a: a5 4a 8e 02  	l.xorix r10, r9, 0xe
   1739e: 01 00        	c.nop
   173a0: 00 00 00 00  	v.nop
   173a4: 00 00 00 00 00 00    	m.nop
   173aa: 69 56 00 12  	l.bc.lz.u32 r25, r10
   173ae: 01 00        	c.nop
   173b0: 00 00 00 00  	v.nop
   173b4: 00 00 00 00 00 00    	m.nop
   173ba: 31 12 db 04  	l.st.w r27, [sp, 0x48]
   173be: 01 00        	c.nop
   173c0: 00 00 00 00  	v.nop
   173c4: 00 00 00 00 00 00    	m.nop
   173ca: bb 69 99 c0  	l.blt.s r13, r25, 0x6 <0x17420>
   173ce: 01 00        	c.nop
   173d0: 00 00 00 00  	v.nop
   173d4: 00 00 00 00 00 00    	m.nop
   173da: 2f 50 80 30  	l.ldi_hi.m1 r10, 0x8000
   173de: 01 00        	c.nop
   173e0: 00 00 00 00  	v.nop
   173e4: 00 00 00 00 00 00    	m.nop
   173ea: a3 52 99 02  	l.srl     r10, r10, r25
   173ee: 01 00        	c.nop
   173f0: 00 00 00 00  	v.nop
   173f4: 00 00 00 00 00 00    	m.nop
   173fa: 65 52 09 02  	l.orix r9, r10, 0x9
   173fe: 01 00        	c.nop
   17400: 00 00 00 00  	v.nop
   17404: 00 00 00 00 00 00    	m.nop
   1740a: 31 58 c9 00  	l.st.w r9, [r11]
   1740e: 01 00        	c.nop
   17410: 00 00 00 00  	v.nop
   17414: 00 00 00 00 00 00    	m.nop
   1741a: 2b 58 80 01  	l.jal r0, 0xb <0x174c0>
   1741e: 01 00        	c.nop
   17420: 00 00 00 00  	v.nop
   17424: 00 00 00 00 00 00    	m.nop
   1742a: b1 5a 44 00  	l.ld.w r10, [r11, 0x4]
   1742e: 01 00        	c.nop
   17430: 00 00 00 00  	v.nop
   17434: 00 00 00 00 00 00    	m.nop
   1743a: 65 52 8e 02  	l.xorix r9, r10, 0xe
   1743e: 01 00        	c.nop
   17440: 00 00 00 00  	v.nop
   17444: 00 00 00 00 00 00    	m.nop
   1744a: 69 4a 00 12  	l.bc.lz.u32 r9, r9
   1744e: 01 00        	c.nop
   17450: 00 00 00 00  	v.nop
   17454: 00 00 00 00 00 00    	m.nop
   1745a: bb 68 89 80  	l.bge.s r13, r9, 0x2 <0x17470>
   1745e: 01 00        	c.nop
   17460: 00 00 00 00  	v.nop
   17464: 00 00 00 00 00 00    	m.nop
   1746a: 2b 40 93 05  	l.jal r0, 0x1138 <0x287e0>
   1746e: 01 00        	c.nop
   17470: 00 00 00 00  	v.nop
   17474: 00 00 00 00 00 00    	m.nop
   1747a: e3 42 89 02  	l.srl     r11, r8, r9
   1747e: 01 00        	c.nop
   17480: 00 00 00 00  	v.nop
   17484: 00 00 00 00 00 00    	m.nop
   1748a: a5 5a 0a 02  	l.orix r10, r11, 0xa
   1748e: 01 00        	c.nop
   17490: 00 00 00 00  	v.nop
   17494: 00 00 00 00 00 00    	m.nop
   1749a: e3 22 05 10  	<unknown>
   1749e: 01 00        	c.nop
   174a0: 00 00 00 00  	v.nop
   174a4: 00 00 00 00 00 00    	m.nop
   174aa: 61 4e 20 00  	l.addi.u r25, r9, 0x20
   174ae: 01 00        	c.nop
   174b0: 00 00 00 00  	v.nop
   174b4: 00 00 00 00 00 00    	m.nop
   174ba: 31 58 ca 00  	l.st.w r10, [r11]
   174be: 01 00        	c.nop
   174c0: 00 00 00 00  	v.nop
   174c4: 00 00 00 00 00 00    	m.nop
   174ca: 2f 70 00 19  	l.ldi16.u r14, 0xc000
   174ce: 85 42        	<unknown>
   174d0: 00 00 00 00  	v.nop
   174d4: 00 00 00 00 00 00    	m.nop
   174da: a3 62 09 03  	l.slli r10, r12, 0x9
   174de: fd 5d        	<unknown>
   174e0: 00 00 00 00  	v.nop
   174e4: 00 00 00 00 00 00    	m.nop
   174ea: 2f 70 80 08  	l.ldi_hi r14, 0x4000
   174ee: 01 00        	c.nop
   174f0: 00 00 00 00  	v.nop
   174f4: 00 00 00 00 00 00    	m.nop
   174fa: 23 52 0e 18  	<unknown>
   174fe: 01 00        	c.nop
   17500: 00 00 00 00  	v.nop
   17504: 00 00 00 00 00 00    	m.nop
   1750a: 2f 51 20 25  	l.ldi16.s r10, 0x2404
   1750e: 01 00        	c.nop
   17510: 00 00 00 00  	v.nop
   17514: 00 00 00 00 00 00    	m.nop
   1751a: 6f 48 80 30  	l.ldi_hi.m1 r9, 0x8001
   1751e: 01 00        	c.nop
   17520: 00 00 00 00  	v.nop
   17524: 00 00 00 00 00 00    	m.nop
   1752a: 31 17 cc 16  	l.st.w r12, [sp, 0x17c]
   1752e: 01 00        	c.nop
   17530: 00 00 00 00  	v.nop
   17534: 00 00 00 00 00 00    	m.nop
   1753a: 31 17 d8 06  	l.st.w r24, [sp, 0x7c]
   1753e: 01 00        	c.nop
   17540: 00 00 00 00  	v.nop
   17544: 00 00 00 00 00 00    	m.nop
   1754a: 2f 5e 04 21  	l.ldi16.s r11, 0x98
   1754e: 01 00        	c.nop
   17550: 00 00 00 00  	v.nop
   17554: 00 00 00 00 00 00    	m.nop
   1755a: ef 3f 7f 1f  	l.ldi16.u r7, 0xffff
   1755e: 01 00        	c.nop
   17560: 00 00 00 00  	v.nop
   17564: 00 00 00 00 00 00    	m.nop
   1756a: 2f 50 80 08  	l.ldi_hi r10, 0x4000
   1756e: 01 00        	c.nop
   17570: 00 00 00 00  	v.nop
   17574: 00 00 00 00 00 00    	m.nop
   1757a: 23 cb 09 18  	<unknown>
   1757e: 01 00        	c.nop
   17580: 00 00 00 00  	v.nop
   17584: 00 00 00 00 00 00    	m.nop
   1758a: 73 03 80 00  	l.movs r13, sip_id
   1758e: 01 00        	c.nop
   17590: 00 00 00 00  	v.nop
   17594: 00 00 00 00 00 00    	m.nop
   1759a: 31 50 cc 00  	l.st.w r12, [r10]
   1759e: 01 00        	c.nop
   175a0: 00 00 00 00  	v.nop
   175a4: 00 00 00 00 00 00    	m.nop
   175aa: 21 3b 02 00  	l.addi.u r12, r7, 0x2
   175ae: 01 00        	c.nop
   175b0: 00 00 00 00  	v.nop
   175b4: 00 00 00 00 00 00    	m.nop
   175ba: 23 46 0b 10  	<unknown>
   175be: 01 00        	c.nop
   175c0: 00 00 00 00  	v.nop
   175c4: 00 00 00 00 00 00    	m.nop
   175ca: 2f 38 40 29  	l.ldi16.s r7, 0x4800
   175ce: 01 00        	c.nop
   175d0: 00 00 00 00  	v.nop
   175d4: 00 00 00 00 00 00    	m.nop
   175da: 73 fb 80 08  	<unknown>
   175de: 01 00        	c.nop
   175e0: 00 00 00 00  	v.nop
   175e4: 00 00 00 00 00 00    	m.nop
   175ea: 73 fb 80 08  	<unknown>
   175ee: 01 00        	c.nop
   175f0: 00 00 00 00  	v.nop
   175f4: 00 00 00 00 00 00    	m.nop
   175fa: 31 17 c8 08  	l.st.w r8, [sp, 0x9c]
   175fe: 01 00        	c.nop
   17600: 00 00 00 00  	v.nop
   17604: 00 00 00 00 00 00    	m.nop
   1760a: 31 40 cc 00  	l.st.w r12, [r8]
   1760e: 01 00        	c.nop
   17610: 00 00 00 00  	v.nop
   17614: 00 00 00 00 00 00    	m.nop
   1761a: 31 c0 c5 00  	l.st.w r5, [r24]
   1761e: 01 00        	c.nop
   17620: 00 00 00 00  	v.nop
   17624: 00 00 00 00 00 00    	m.nop
   1762a: 31 c6 c5 fe  	l.st.w r5, [r24, -0x8]
   1762e: 01 00        	c.nop
   17630: 00 00 00 00  	v.nop
   17634: 00 00 00 00 00 00    	m.nop
   1763a: 31 c2 c5 00  	l.st.w r5, [r24, 0x8]
   1763e: 01 00        	c.nop
   17640: 00 00 00 00  	v.nop
   17644: 00 00 00 00 00 00    	m.nop
   1764a: 31 13 d9 06  	l.st.w r25, [sp, 0x6c]
   1764e: 01 00        	c.nop
   17650: 00 00 00 00  	v.nop
   17654: 00 00 00 00 00 00    	m.nop
   1765a: 31 c3 d9 00  	l.st.w r25, [r24, 0xc]
   1765e: 01 00        	c.nop
   17660: 00 00 00 00  	v.nop
   17664: 00 00 00 00 00 00    	m.nop
   1766a: 31 c2 c7 00  	l.st.w r7, [r24, 0x8]
   1766e: 01 00        	c.nop
   17670: 00 00 00 00  	v.nop
   17674: 00 00 00 00 00 00    	m.nop
   1767a: 31 26 40 00  	l.ld.w r24, [r4]
   1767e: 01 00        	c.nop
   17680: 00 00 00 00  	v.nop
   17684: 00 00 00 00 00 00    	m.nop
   1768a: 2f 40 01 21  	l.ldi16.s r8, 0x20
   1768e: 01 00        	c.nop
   17690: 00 00 00 00  	v.nop
   17694: 00 00 00 00 00 00    	m.nop
   1769a: 65 c6 9b 02  	l.xorix r25, r24, 0x1b
   1769e: 01 00        	c.nop
   176a0: 00 00 00 00  	v.nop
   176a4: 00 00 00 00 00 00    	m.nop
   176aa: 69 ce 00 12  	l.bc.lz.u32 r25, r25
   176ae: 01 00        	c.nop
   176b0: 00 00 00 00  	v.nop
   176b4: 00 00 00 00 00 00    	m.nop
   176ba: bb c8 88 c0  	l.blt.s r25, r8, 0x2 <0x176d0>
   176be: 01 00        	c.nop
   176c0: 00 00 00 00  	v.nop
   176c4: 00 00 00 00 00 00    	m.nop
   176ca: 2b 10 91 05  	l.jal r0, 0x1112 <0x287e0>
   176ce: 01 00        	c.nop
   176d0: 00 00 00 00  	v.nop
   176d4: 00 00 00 00 00 00    	m.nop
   176da: 2f 40 80 30  	l.ldi_hi.m1 r8, 0x8000
   176de: 7d 4d        	<unknown>
   176e0: 00 00 00 00  	v.nop
   176e4: 00 00 00 00 00 00    	m.nop
   176ea: 23 42 99 02  	l.srl     r8, r8, r25
   176ee: 01 00        	c.nop
   176f0: 00 00 00 00  	v.nop
   176f4: 00 00 00 00 00 00    	m.nop
   176fa: 25 46 18 02  	l.orix r24, r8, 0x18
   176fe: 01 00        	c.nop
   17700: 00 00 00 00  	v.nop
   17704: 00 00 00 00 00 00    	m.nop
   1770a: 31 14 c8 16  	l.st.w r8, [sp, 0x170]
   1770e: 01 00        	c.nop
   17710: 00 00 00 00  	v.nop
   17714: 00 00 00 00 00 00    	m.nop
   1771a: 31 12 4c 04  	l.ld.w r8, [sp, 0x4c]
   1771e: 01 00        	c.nop
   17720: 00 00 00 00  	v.nop
   17724: 00 00 00 00 00 00    	m.nop
   1772a: 31 20 d8 00  	l.st.w r24, [r4]
   1772e: 01 00        	c.nop
   17730: 00 00 00 00  	v.nop
   17734: 00 00 00 00 00 00    	m.nop
   1773a: 31 17 d9 0c  	l.st.w r25, [sp, 0xdc]
   1773e: 01 00        	c.nop
   17740: 00 00 00 00  	v.nop
   17744: 00 00 00 00 00 00    	m.nop
   1774a: 31 46 40 00  	l.ld.w r24, [r8]
   1774e: 01 00        	c.nop
   17750: 00 00 00 00  	v.nop
   17754: 00 00 00 00 00 00    	m.nop
   1775a: 65 c6 9b 02  	l.xorix r25, r24, 0x1b
   1775e: 01 00        	c.nop
   17760: 00 00 00 00  	v.nop
   17764: 00 00 00 00 00 00    	m.nop
   1776a: a9 cf 00 12  	l.bc.lz.u32 r30, r25
   1776e: 01 00        	c.nop
   17770: 00 00 00 00  	v.nop
   17774: 00 00 00 00 00 00    	m.nop
   1777a: bb d1 9e c0  	l.blt.s r26, r30, 0x6 <0x177d0>
   1777e: 01 00        	c.nop
   17780: 00 00 00 00  	v.nop
   17784: 00 00 00 00 00 00    	m.nop
   1778a: 2f c8 80 30  	l.ldi_hi.m1 r25, 0x8000
   1778e: a1 46        	<unknown>
   17790: 00 00 00 00  	v.nop
   17794: 00 00 00 00 00 00    	m.nop
   1779a: 63 ce 9e 02  	l.srl     r25, r25, r30
   1779e: 01 00        	c.nop
   177a0: 00 00 00 00  	v.nop
   177a4: 00 00 00 00 00 00    	m.nop
   177aa: 25 ce 18 02  	l.orix r24, r25, 0x18
   177ae: 01 00        	c.nop
   177b0: 00 00 00 00  	v.nop
   177b4: 00 00 00 00 00 00    	m.nop
   177ba: 31 40 d8 00  	l.st.w r24, [r8]
   177be: 01 00        	c.nop
   177c0: 00 00 00 00  	v.nop
   177c4: 00 00 00 00 00 00    	m.nop
   177ca: 2b 60 80 01  	l.jal r0, 0xc <0x17880>
   177ce: 01 00        	c.nop
   177d0: 00 00 00 00  	v.nop
   177d4: 00 00 00 00 00 00    	m.nop
   177da: 71 46 44 00  	l.ld.w r25, [r8, 0x4]
   177de: 01 00        	c.nop
   177e0: 00 00 00 00  	v.nop
   177e4: 00 00 00 00 00 00    	m.nop
   177ea: 25 ce 9b 02  	l.xorix r24, r25, 0x1b
   177ee: 01 00        	c.nop
   177f0: 00 00 00 00  	v.nop
   177f4: 00 00 00 00 00 00    	m.nop
   177fa: 29 c6 00 12  	l.bc.lz.u32 r24, r24
   177fe: 01 00        	c.nop
   17800: 00 00 00 00  	v.nop
   17804: 00 00 00 00 00 00    	m.nop
   1780a: bb d0 98 80  	l.bge.s r26, r24, 0x2 <0x17820>
   1780e: 01 00        	c.nop
   17810: 00 00 00 00  	v.nop
   17814: 00 00 00 00 00 00    	m.nop
   1781a: 2b 68 8f 05  	l.jal r0, 0x10fd <0x287e0>
   1781e: 01 00        	c.nop
   17820: 00 00 00 00  	v.nop
   17824: 00 00 00 00 00 00    	m.nop
   1782a: 2f 40 80 30  	l.ldi_hi.m1 r8, 0x8000
   1782e: a1 46        	<unknown>
   17830: 00 00 00 00  	v.nop
   17834: 00 00 00 00 00 00    	m.nop
   1783a: a3 46 98 02  	l.srl     r26, r8, r24
   1783e: 01 00        	c.nop
   17840: 00 00 00 00  	v.nop
   17844: 00 00 00 00 00 00    	m.nop
   1784a: 65 d6 19 02  	l.orix r25, r26, 0x19
   1784e: 01 00        	c.nop
   17850: 00 00 00 00  	v.nop
   17854: 00 00 00 00 00 00    	m.nop
   1785a: a3 26 0d 10  	<unknown>
   1785e: 01 00        	c.nop
   17860: 00 00 00 00  	v.nop
   17864: 00 00 00 00 00 00    	m.nop
   1786a: a1 c7 20 00  	l.addi.u r30, r24, 0x20
   1786e: 01 00        	c.nop
   17870: 00 00 00 00  	v.nop
   17874: 00 00 00 00 00 00    	m.nop
   1787a: 31 d0 d9 00  	l.st.w r25, [r26]
   1787e: 01 00        	c.nop
   17880: 00 00 00 00  	v.nop
   17884: 00 00 00 00 00 00    	m.nop
   1788a: 71 11 5c 02  	l.ld.w r5, [sp, 0x3c]
   1788e: 01 00        	c.nop
   17890: 00 00 00 00  	v.nop
   17894: 00 00 00 00 00 00    	m.nop
   1789a: 23 9e 90 03  	l.srli r24, r19, 0x10
   1789e: 01 00        	c.nop
   178a0: 00 00 00 00  	v.nop
   178a4: 00 00 00 00 00 00    	m.nop
   178aa: 25 c6 9c 01  	l.andix r24, r24, 0x1c
   178ae: 01 00        	c.nop
   178b0: 00 00 00 00  	v.nop
   178b4: 00 00 00 00 00 00    	m.nop
   178ba: 31 12 5c 0c  	l.ld.w r8, [sp, 0xdc]
   178be: 01 00        	c.nop
   178c0: 00 00 00 00  	v.nop
   178c4: 00 00 00 00 00 00    	m.nop
   178ca: 65 29 17 1b  	<unknown>
   178ce: 01 00        	c.nop
   178d0: 00 00 00 00  	v.nop
   178d4: 00 00 00 00 00 00    	m.nop
   178da: e3 9d 98 03  	l.srli r23, r19, 0x18
   178de: 01 00        	c.nop
   178e0: 00 00 00 00  	v.nop
   178e4: 00 00 00 00 00 00    	m.nop
   178ea: e3 b5 17 19  	<unknown>
   178ee: 01 00        	c.nop
   178f0: 00 00 00 00  	v.nop
   178f4: 00 00 00 00 00 00    	m.nop
   178fa: a3 b5 15 19  	<unknown>
   178fe: 01 00        	c.nop
   17900: 00 00 00 00  	v.nop
   17904: 00 00 00 00 00 00    	m.nop
   1790a: 00 00        	s.nop16
   1790c: 00 00        	s.nop16
   1790e: e2 9b        	<unknown>
   17910: 00 00 00 00  	v.nop
   17914: 00 00 00 00 00 00    	m.nop
   1791a: 63 bd 15 19  	<unknown>
   1791e: 01 00        	c.nop
   17920: 00 00 00 00  	v.nop
   17924: 00 00 00 00 00 00    	m.nop
   1792a: e3 8d 98 03  	l.srli r23, r17, 0x18
   1792e: 01 00        	c.nop
   17930: 00 00 00 00  	v.nop
   17934: 00 00 00 00 00 00    	m.nop
   1793a: 31 16 c5 18  	l.st.w r5, [sp, 0x198]
   1793e: 01 00        	c.nop
   17940: 00 00 00 00  	v.nop
   17944: 00 00 00 00 00 00    	m.nop
   1794a: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   1794e: 01 00        	c.nop
   17950: 00 00 00 00  	v.nop
   17954: 00 00 00 00 00 00    	m.nop
   1795a: e5 9c 85 01  	l.andix r19, r19, 0x5
   1795e: 01 00        	c.nop
   17960: 00 00 00 00  	v.nop
   17964: 00 00 00 00 00 00    	m.nop
   1796a: e3 ac 13 18  	<unknown>
   1796e: 01 00        	c.nop
   17970: 00 00 00 00  	v.nop
   17974: 00 00 00 00 00 00    	m.nop
   1797a: e3 94 13 19  	<unknown>
   1797e: 01 00        	c.nop
   17980: 00 00 00 00  	v.nop
   17984: 00 00 00 00 00 00    	m.nop
   1798a: 63 b5 10 19  	<unknown>
   1798e: 01 00        	c.nop
   17990: 00 00 00 00  	v.nop
   17994: 00 00 00 00 00 00    	m.nop
   1799a: a3 8d 90 03  	l.srli r22, r17, 0x10
   1799e: 01 00        	c.nop
   179a0: 00 00 00 00  	v.nop
   179a4: 00 00 00 00 00 00    	m.nop
   179aa: 65 8c 85 01  	l.andix r17, r17, 0x5
   179ae: de 99        	<unknown>
   179b0: 00 00 00 00  	v.nop
   179b4: 00 00 00 00 00 00    	m.nop
   179ba: e3 9c 14 19  	<unknown>
   179be: 85 42        	<unknown>
   179c0: 00 00 00 00  	v.nop
   179c4: 00 00 00 00 00 00    	m.nop
   179ca: 23 ad 14 19  	<unknown>
   179ce: 01 00        	c.nop
   179d0: 00 00 00 00  	v.nop
   179d4: 00 00 00 00 00 00    	m.nop
   179da: 65 b5 9c 01  	l.andix r21, r22, 0x1c
   179de: 01 00        	c.nop
   179e0: 00 00 00 00  	v.nop
   179e4: 00 00 00 00 00 00    	m.nop
   179ea: 2f a8 00 2d  	l.ldi16.s r21, 0x6000
   179ee: d6 99        	<unknown>
   179f0: 00 00 00 00  	v.nop
   179f4: 00 00 00 00 00 00    	m.nop
   179fa: a3 a4 12 19  	<unknown>
   179fe: 01 00        	c.nop
   17a00: 00 00 00 00  	v.nop
   17a04: 00 00 00 00 00 00    	m.nop
   17a0a: 23 45 09 03  	l.slli r20, r8, 0x9
   17a0e: 01 00        	c.nop
   17a10: 00 00 00 00  	v.nop
   17a14: 00 00 00 00 00 00    	m.nop
   17a1a: 23 9c 10 19  	<unknown>
   17a1e: 01 00        	c.nop
   17a20: 00 00 00 00  	v.nop
   17a24: 00 00 00 00 00 00    	m.nop
   17a2a: 23 a2 0e 18  	<unknown>
   17a2e: 01 00        	c.nop
   17a30: 00 00 00 00  	v.nop
   17a34: 00 00 00 00 00 00    	m.nop
   17a3a: e3 f4 09 18  	<unknown>
   17a3e: 01 00        	c.nop
   17a40: 00 00 00 00  	v.nop
   17a44: 00 00 00 00 00 00    	m.nop
   17a4a: 33 05 80 00  	l.movs r20, sip_id
   17a4e: 01 00        	c.nop
   17a50: 00 00 00 00  	v.nop
   17a54: 00 00 00 00 00 00    	m.nop
   17a5a: 31 50 d3 00  	l.st.w r19, [r10]
   17a5e: 01 00        	c.nop
   17a60: 00 00 00 00  	v.nop
   17a64: 00 00 00 00 00 00    	m.nop
   17a6a: e3 44 0b 10  	<unknown>
   17a6e: 01 00        	c.nop
   17a70: 00 00 00 00  	v.nop
   17a74: 00 00 00 00 00 00    	m.nop
   17a7a: 33 fd 80 08  	<unknown>
   17a7e: 01 00        	c.nop
   17a80: 00 00 00 00  	v.nop
   17a84: 00 00 00 00 00 00    	m.nop
   17a8a: 33 fd 80 08  	<unknown>
   17a8e: 01 00        	c.nop
   17a90: 00 00 00 00  	v.nop
   17a94: 00 00 00 00 00 00    	m.nop
   17a9a: 31 12 c8 0a  	l.st.w r8, [sp, 0xa8]
   17a9e: 01 00        	c.nop
   17aa0: 00 00 00 00  	v.nop
   17aa4: 00 00 00 00 00 00    	m.nop
   17aaa: 31 40 cc 00  	l.st.w r12, [r8]
   17aae: 01 00        	c.nop
   17ab0: 00 00 00 00  	v.nop
   17ab4: 00 00 00 00 00 00    	m.nop
   17aba: 31 98 c5 00  	l.st.w r5, [r19]
   17abe: 01 00        	c.nop
   17ac0: 00 00 00 00  	v.nop
   17ac4: 00 00 00 00 00 00    	m.nop
   17aca: 31 9e c5 fe  	l.st.w r5, [r19, -0x8]
   17ace: 01 00        	c.nop
   17ad0: 00 00 00 00  	v.nop
   17ad4: 00 00 00 00 00 00    	m.nop
   17ada: 31 9a c5 00  	l.st.w r5, [r19, 0x8]
   17ade: 01 00        	c.nop
   17ae0: 00 00 00 00  	v.nop
   17ae4: 00 00 00 00 00 00    	m.nop
   17aea: 63 81 11 18  	<unknown>
   17aee: 01 00        	c.nop
   17af0: 00 00 00 00  	v.nop
   17af4: 00 00 00 00 00 00    	m.nop
   17afa: 31 9b de 00  	l.st.w r30, [r19, 0xc]
   17afe: 01 00        	c.nop
   17b00: 00 00 00 00  	v.nop
   17b04: 00 00 00 00 00 00    	m.nop
   17b0a: 31 11 c5 06  	l.st.w r5, [sp, 0x64]
   17b0e: 01 00        	c.nop
   17b10: 00 00 00 00  	v.nop
   17b14: 00 00 00 00 00 00    	m.nop
   17b1a: 63 91 0f 19  	<unknown>
   17b1e: 01 00        	c.nop
   17b20: 00 00 00 00  	v.nop
   17b24: 00 00 00 00 00 00    	m.nop
   17b2a: 31 12 50 08  	l.ld.w r8, [sp, 0x90]
   17b2e: 01 00        	c.nop
   17b30: 00 00 00 00  	v.nop
   17b34: 00 00 00 00 00 00    	m.nop
   17b3a: 2f 78 00 2d  	l.ldi16.s r15, 0x6000
   17b3e: 01 00        	c.nop
   17b40: 00 00 00 00  	v.nop
   17b44: 00 00 00 00 00 00    	m.nop
   17b4a: 31 12 c5 1a  	l.st.w r5, [sp, 0x1a8]
   17b4e: 01 00        	c.nop
   17b50: 00 00 00 00  	v.nop
   17b54: 00 00 00 00 00 00    	m.nop
   17b5a: 71 11 5c 06  	l.ld.w r5, [sp, 0x7c]
   17b5e: 01 00        	c.nop
   17b60: 00 00 00 00  	v.nop
   17b64: 00 00 00 00 00 00    	m.nop
   17b6a: 2f 80 40 29  	l.ldi16.s r16, 0x4800
   17b6e: 01 00        	c.nop
   17b70: 00 00 00 00  	v.nop
   17b74: 00 00 00 00 00 00    	m.nop
   17b7a: af 7f 91 08  	l.ldi_hi r15, 0x423e
   17b7e: 01 00        	c.nop
   17b80: 00 00 00 00  	v.nop
   17b84: 00 00 00 00 00 00    	m.nop
   17b8a: 2f 80 a0 00  	l.ldi_hi r16, 0x400
   17b8e: 01 00        	c.nop
   17b90: 00 00 00 00  	v.nop
   17b94: 00 00 00 00 00 00    	m.nop
   17b9a: af 8f 81 20  	l.ldi_hi.m1 r17, 0x3e
   17b9e: 01 00        	c.nop
   17ba0: 00 00 00 00  	v.nop
   17ba4: 00 00 00 00 00 00    	m.nop
   17baa: 63 41 85 18  	<unknown>
   17bae: 01 00        	c.nop
   17bb0: 00 00 00 00  	v.nop
   17bb4: 00 00 00 00 00 00    	m.nop
   17bba: 31 16 c5 08  	l.st.w r5, [sp, 0x98]
   17bbe: 01 00        	c.nop
   17bc0: 00 00 00 00  	v.nop
   17bc4: 00 00 00 00 00 00    	m.nop
   17bca: 31 9a c7 00  	l.st.w r7, [r19, 0x8]
   17bce: 01 00        	c.nop
   17bd0: 00 00 00 00  	v.nop
   17bd4: 00 00 00 00 00 00    	m.nop
   17bda: 63 21 0d 10  	<unknown>
   17bde: 01 00        	c.nop
   17be0: 00 00 00 00  	v.nop
   17be4: 00 00 00 00 00 00    	m.nop
   17bea: 2f b8 29 21  	l.ldi16.s r23, 0x520
   17bee: 01 00        	c.nop
   17bf0: 00 00 00 00  	v.nop
   17bf4: 00 00 00 00 00 00    	m.nop
   17bfa: 2f 91 05 21  	l.ldi16.s r18, 0xa4
   17bfe: 01 00        	c.nop
   17c00: 00 00 00 00  	v.nop
   17c04: 00 00 00 00 00 00    	m.nop
   17c0a: ef 9f 0f 21  	l.ldi16.s r19, 0x1ff
   17c0e: 01 00        	c.nop
   17c10: 00 00 00 00  	v.nop
   17c14: 00 00 00 00 00 00    	m.nop
   17c1a: 2f a1 08 21  	l.ldi16.s r20, 0x104
   17c1e: 01 00        	c.nop
   17c20: 00 00 00 00  	v.nop
   17c24: 00 00 00 00 00 00    	m.nop
   17c2a: af ae a3 00  	l.ldi_hi r21, 0x47a
   17c2e: 01 00        	c.nop
   17c30: 00 00 00 00  	v.nop
   17c34: 00 00 00 00 00 00    	m.nop
   17c3a: 2f b4 04 21  	l.ldi16.s r22, 0x90
   17c3e: 01 00        	c.nop
   17c40: 00 00 00 00  	v.nop
   17c44: 00 00 00 00 00 00    	m.nop
   17c4a: 31 11 c5 02  	l.st.w r5, [sp, 0x24]
   17c4e: 01 00        	c.nop
   17c50: 00 00 00 00  	v.nop
   17c54: 00 00 00 00 00 00    	m.nop
   17c5a: 2b 50 80 01  	l.jal r0, 0xa <0x17cf0>
   17c5e: 01 00        	c.nop
   17c60: 00 00 00 00  	v.nop
   17c64: 00 00 00 00 00 00    	m.nop
   17c6a: 31 c1 54 72  	l.ld.w r4, [r24, 0x734]
   17c6e: 01 00        	c.nop
   17c70: 00 00 00 00  	v.nop
   17c74: 00 00 00 00 00 00    	m.nop
   17c7a: 23 da 1c 18  	<unknown>
   17c7e: 01 00        	c.nop
   17c80: 00 00 00 00  	v.nop
   17c84: 00 00 00 00 00 00    	m.nop
   17c8a: 00 00        	s.nop16
   17c8c: 00 00        	s.nop16
   17c8e: 4a 94        	<unknown>
   17c90: 00 00 00 00  	v.nop
   17c94: 00 00 00 00 00 00    	m.nop
   17c9a: 31 40 c4 00  	l.st.w r4, [r8]
   17c9e: 01 00        	c.nop
   17ca0: 00 00 00 00  	v.nop
   17ca4: 00 00 00 00 00 00    	m.nop
   17caa: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   17cae: 01 00        	c.nop
   17cb0: 00 00 00 00  	v.nop
   17cb4: 00 00 00 00 00 00    	m.nop
   17cba: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   17cbe: 01 00        	c.nop
   17cc0: 00 00 00 00  	v.nop
   17cc4: 00 00 00 00 00 00    	m.nop
   17cca: 31 47 d9 fe  	l.st.w r25, [r8, -0x4]
   17cce: 01 00        	c.nop
   17cd0: 00 00 00 00  	v.nop
   17cd4: 00 00 00 00 00 00    	m.nop
   17cda: e1 bd 70 2d  	l.addi.s r23, r23, -0x290
   17cde: 01 00        	c.nop
   17ce0: 00 00 00 00  	v.nop
   17ce4: 00 00 00 00 00 00    	m.nop
   17cea: 7b b8 40 02  	<unknown>
   17cee: 01 00        	c.nop
   17cf0: 00 00 00 00  	v.nop
   17cf4: 00 00 00 00 00 00    	m.nop
   17cfa: 67 11 a2 09  	l.addi_sft.s.sft0 r5, sp, 0x22
   17cfe: 81 46        	<unknown>
   17d00: 00 00 00 00  	v.nop
   17d04: 00 00 00 00 00 00    	m.nop
   17d0a: 61 29 00 20  	l.addi.s r5, r5, 0x0
   17d0e: fd 5e        	<unknown>
   17d10: 00 00 00 00  	v.nop
   17d14: 00 00 00 00 00 00    	m.nop
   17d1a: 23 2e 97 18  	<unknown>
   17d1e: 01 00        	c.nop
   17d20: 00 00 00 00  	v.nop
   17d24: 00 00 00 00 00 00    	m.nop
   17d2a: 31 c1 cd 52  	l.st.w r13, [r24, 0x524]
   17d2e: 01 00        	c.nop
   17d30: 00 00 00 00  	v.nop
   17d34: 00 00 00 00 00 00    	m.nop
   17d3a: b1 26 40 00  	l.ld.w r26, [r4]
   17d3e: 01 00        	c.nop
   17d40: 00 00 00 00  	v.nop
   17d44: 00 00 00 00 00 00    	m.nop
   17d4a: 31 17 c5 12  	l.st.w r5, [sp, 0x13c]
   17d4e: 85 42        	<unknown>
   17d50: 00 00 00 00  	v.nop
   17d54: 00 00 00 00 00 00    	m.nop
   17d5a: 2f 40 01 21  	l.ldi16.s r8, 0x20
   17d5e: 01 00        	c.nop
   17d60: 00 00 00 00  	v.nop
   17d64: 00 00 00 00 00 00    	m.nop
   17d6a: b1 c2 8d 52  	l.st.b r13, [r24, 0x52a]
   17d6e: 01 00        	c.nop
   17d70: 00 00 00 00  	v.nop
   17d74: 00 00 00 00 00 00    	m.nop
   17d7a: 31 c2 85 52  	l.st.b r5, [r24, 0x528]
   17d7e: 01 00        	c.nop
   17d80: 00 00 00 00  	v.nop
   17d84: 00 00 00 00 00 00    	m.nop
   17d8a: 65 d6 9d 02  	l.xorix r25, r26, 0x1d
   17d8e: 01 00        	c.nop
   17d90: 00 00 00 00  	v.nop
   17d94: 00 00 00 00 00 00    	m.nop
   17d9a: 69 ce 00 12  	l.bc.lz.u32 r25, r25
   17d9e: 01 00        	c.nop
   17da0: 00 00 00 00  	v.nop
   17da4: 00 00 00 00 00 00    	m.nop
   17daa: bb c8 88 c0  	l.blt.s r25, r8, 0x2 <0x17dc0>
   17dae: 01 00        	c.nop
   17db0: 00 00 00 00  	v.nop
   17db4: 00 00 00 00 00 00    	m.nop
   17dba: 2b 18 8a 05  	l.jal r0, 0x10a3 <0x287e0>
   17dbe: 01 00        	c.nop
   17dc0: 00 00 00 00  	v.nop
   17dc4: 00 00 00 00 00 00    	m.nop
   17dca: 2f 40 80 30  	l.ldi_hi.m1 r8, 0x8000
   17dce: 01 00        	c.nop
   17dd0: 00 00 00 00  	v.nop
   17dd4: 00 00 00 00 00 00    	m.nop
   17dda: 31 17 4c 04  	l.ld.w r28, [sp, 0x4c]
   17dde: 01 00        	c.nop
   17de0: 00 00 00 00  	v.nop
   17de4: 00 00 00 00 00 00    	m.nop
   17dea: e3 46 99 02  	l.srl     r27, r8, r25
   17dee: 01 00        	c.nop
   17df0: 00 00 00 00  	v.nop
   17df4: 00 00 00 00 00 00    	m.nop
   17dfa: a5 de 1a 02  	l.orix r26, r27, 0x1a
   17dfe: 01 00        	c.nop
   17e00: 00 00 00 00  	v.nop
   17e04: 00 00 00 00 00 00    	m.nop
   17e0a: e3 ce 01 03  	l.slli r27, r25, 0x1
   17e0e: 01 00        	c.nop
   17e10: 00 00 00 00  	v.nop
   17e14: 00 00 00 00 00 00    	m.nop
   17e1a: 31 20 da 00  	l.st.w r26, [r4]
   17e1e: 01 00        	c.nop
   17e20: 00 00 00 00  	v.nop
   17e24: 00 00 00 00 00 00    	m.nop
   17e2a: a1 de 01 00  	l.addi.u r26, r27, 0x1
   17e2e: 01 00        	c.nop
   17e30: 00 00 00 00  	v.nop
   17e34: 00 00 00 00 00 00    	m.nop
   17e3a: 31 c7 cd 5e  	l.st.w r13, [r24, 0x5fc]
   17e3e: 01 00        	c.nop
   17e40: 00 00 00 00  	v.nop
   17e44: 00 00 00 00 00 00    	m.nop
   17e4a: 31 c2 cd 62  	l.st.w r13, [r24, 0x628]
   17e4e: 01 00        	c.nop
   17e50: 00 00 00 00  	v.nop
   17e54: 00 00 00 00 00 00    	m.nop
   17e5a: 31 c5 cd 5c  	l.st.w r13, [r24, 0x5d4]
   17e5e: 01 00        	c.nop
   17e60: 00 00 00 00  	v.nop
   17e64: 00 00 00 00 00 00    	m.nop
   17e6a: 31 c0 cd 5e  	l.st.w r13, [r24, 0x5e0]
   17e6e: 01 00        	c.nop
   17e70: 00 00 00 00  	v.nop
   17e74: 00 00 00 00 00 00    	m.nop
   17e7a: 31 c5 cd 52  	l.st.w r13, [r24, 0x534]
   17e7e: 01 00        	c.nop
   17e80: 00 00 00 00  	v.nop
   17e84: 00 00 00 00 00 00    	m.nop
   17e8a: 31 c6 d9 52  	l.st.w r25, [r24, 0x538]
   17e8e: 01 00        	c.nop
   17e90: 00 00 00 00  	v.nop
   17e94: 00 00 00 00 00 00    	m.nop
   17e9a: 31 c0 d9 52  	l.st.w r25, [r24, 0x520]
   17e9e: a6 0c        	<unknown>
   17ea0: 00 00 00 00  	v.nop
   17ea4: 00 00 00 00 00 00    	m.nop
   17eaa: 31 c0 da 54  	l.st.w r26, [r24, 0x540]
   17eae: ba 9c        	<unknown>
   17eb0: 00 00 00 00  	v.nop
   17eb4: 00 00 00 00 00 00    	m.nop
   17eba: 31 c7 db 52  	l.st.w r27, [r24, 0x53c]
   17ebe: 01 00        	c.nop
   17ec0: 00 00 00 00  	v.nop
   17ec4: 00 00 00 00 00 00    	m.nop
   17eca: 31 c1 cd 54  	l.st.w r13, [r24, 0x544]
   17ece: 01 00        	c.nop
   17ed0: 00 00 00 00  	v.nop
   17ed4: 00 00 00 00 00 00    	m.nop
   17eda: 31 c2 ad 54  	l.st.h r13, [r24, 0x548]
   17ede: 01 00        	c.nop
   17ee0: 00 00 00 00  	v.nop
   17ee4: 00 00 00 00 00 00    	m.nop
   17eea: 31 c3 dd 54  	l.st.w r29, [r24, 0x54c]
   17eee: 01 00        	c.nop
   17ef0: 00 00 00 00  	v.nop
   17ef4: 00 00 00 00 00 00    	m.nop
   17efa: 31 c4 dd 54  	l.st.w r29, [r24, 0x550]
   17efe: 01 00        	c.nop
   17f00: 00 00 00 00  	v.nop
   17f04: 00 00 00 00 00 00    	m.nop
   17f0a: 31 c5 dd 54  	l.st.w r29, [r24, 0x554]
   17f0e: 01 00        	c.nop
   17f10: 00 00 00 00  	v.nop
   17f14: 00 00 00 00 00 00    	m.nop
   17f1a: 31 c6 cd 54  	l.st.w r13, [r24, 0x558]
   17f1e: 01 00        	c.nop
   17f20: 00 00 00 00  	v.nop
   17f24: 00 00 00 00 00 00    	m.nop
   17f2a: 31 c7 cd 54  	l.st.w r13, [r24, 0x55c]
   17f2e: 01 00        	c.nop
   17f30: 00 00 00 00  	v.nop
   17f34: 00 00 00 00 00 00    	m.nop
   17f3a: 31 c1 dd 56  	l.st.w r29, [r24, 0x564]
   17f3e: 01 00        	c.nop
   17f40: 00 00 00 00  	v.nop
   17f44: 00 00 00 00 00 00    	m.nop
   17f4a: 31 c2 dd 56  	l.st.w r29, [r24, 0x568]
   17f4e: 01 00        	c.nop
   17f50: 00 00 00 00  	v.nop
   17f54: 00 00 00 00 00 00    	m.nop
   17f5a: 31 c3 dd 56  	l.st.w r29, [r24, 0x56c]
   17f5e: 01 00        	c.nop
   17f60: 00 00 00 00  	v.nop
   17f64: 00 00 00 00 00 00    	m.nop
   17f6a: 31 c4 cd 56  	l.st.w r13, [r24, 0x570]
   17f6e: 01 00        	c.nop
   17f70: 00 00 00 00  	v.nop
   17f74: 00 00 00 00 00 00    	m.nop
   17f7a: 31 c5 cd 56  	l.st.w r13, [r24, 0x574]
   17f7e: 01 00        	c.nop
   17f80: 00 00 00 00  	v.nop
   17f84: 00 00 00 00 00 00    	m.nop
   17f8a: 31 c6 cd 56  	l.st.w r13, [r24, 0x578]
   17f8e: 01 00        	c.nop
   17f90: 00 00 00 00  	v.nop
   17f94: 00 00 00 00 00 00    	m.nop
   17f9a: 31 c7 cd 56  	l.st.w r13, [r24, 0x57c]
   17f9e: 01 00        	c.nop
   17fa0: 00 00 00 00  	v.nop
   17fa4: 00 00 00 00 00 00    	m.nop
   17faa: 31 c1 cd 58  	l.st.w r13, [r24, 0x584]
   17fae: 01 00        	c.nop
   17fb0: 00 00 00 00  	v.nop
   17fb4: 00 00 00 00 00 00    	m.nop
   17fba: 31 c2 cd 58  	l.st.w r13, [r24, 0x588]
   17fbe: 01 00        	c.nop
   17fc0: 00 00 00 00  	v.nop
   17fc4: 00 00 00 00 00 00    	m.nop
   17fca: 31 c4 ad 5a  	l.st.h r13, [r24, 0x5b0]
   17fce: 01 00        	c.nop
   17fd0: 00 00 00 00  	v.nop
   17fd4: 00 00 00 00 00 00    	m.nop
   17fda: 31 c0 dd 72  	l.st.w r29, [r24, 0x720]
   17fde: 01 00        	c.nop
   17fe0: 00 00 00 00  	v.nop
   17fe4: 00 00 00 00 00 00    	m.nop
   17fea: b1 e6 40 00  	l.ld.w r26, [r28]
   17fee: 01 00        	c.nop
   17ff0: 00 00 00 00  	v.nop
   17ff4: 00 00 00 00 00 00    	m.nop
   17ffa: e3 ce 9f 0b  	l.srai r27, r25, 0x1f
   17ffe: 01 00        	c.nop
   18000: 00 00 00 00  	v.nop
   18004: 00 00 00 00 00 00    	m.nop
   1800a: 31 c6 d9 5a  	l.st.w r25, [r24, 0x5b8]
   1800e: 01 00        	c.nop
   18010: 00 00 00 00  	v.nop
   18014: 00 00 00 00 00 00    	m.nop
   1801a: 31 c7 db 5a  	l.st.w r27, [r24, 0x5bc]
   1801e: fd 4d        	<unknown>
   18020: 00 00 00 00  	v.nop
   18024: 00 00 00 00 00 00    	m.nop
   1802a: 31 c6 8d 70  	l.st.b r13, [r24, 0x718]
   1802e: 01 00        	c.nop
   18030: 00 00 00 00  	v.nop
   18034: 00 00 00 00 00 00    	m.nop
   1803a: 65 d6 9d 02  	l.xorix r25, r26, 0x1d
   1803e: 01 00        	c.nop
   18040: 00 00 00 00  	v.nop
   18044: 00 00 00 00 00 00    	m.nop
   1804a: 69 ce 00 12  	l.bc.lz.u32 r25, r25
   1804e: 01 00        	c.nop
   18050: 00 00 00 00  	v.nop
   18054: 00 00 00 00 00 00    	m.nop
   1805a: 31 c6 c5 5c  	l.st.w r5, [r24, 0x5d8]
   1805e: 01 00        	c.nop
   18060: 00 00 00 00  	v.nop
   18064: 00 00 00 00 00 00    	m.nop
   1806a: 31 c7 c5 5c  	l.st.w r5, [r24, 0x5dc]
   1806e: 01 00        	c.nop
   18070: 00 00 00 00  	v.nop
   18074: 00 00 00 00 00 00    	m.nop
   1807a: 31 c1 c5 5e  	l.st.w r5, [r24, 0x5e4]
   1807e: 01 00        	c.nop
   18080: 00 00 00 00  	v.nop
   18084: 00 00 00 00 00 00    	m.nop
   1808a: 31 c2 c5 5e  	l.st.w r5, [r24, 0x5e8]
   1808e: 01 00        	c.nop
   18090: 00 00 00 00  	v.nop
   18094: 00 00 00 00 00 00    	m.nop
   1809a: 31 c4 c5 52  	l.st.w r5, [r24, 0x530]
   1809e: 01 00        	c.nop
   180a0: 00 00 00 00  	v.nop
   180a4: 00 00 00 00 00 00    	m.nop
   180aa: 31 c0 8d 56  	l.st.b r13, [r24, 0x560]
   180ae: 01 00        	c.nop
   180b0: 00 00 00 00  	v.nop
   180b4: 00 00 00 00 00 00    	m.nop
   180ba: 71 c3 8d 58  	l.st.b r13, [r24, 0x58d]
   180be: 01 00        	c.nop
   180c0: 00 00 00 00  	v.nop
   180c4: 00 00 00 00 00 00    	m.nop
   180ca: 31 c0 8d 58  	l.st.b r13, [r24, 0x580]
   180ce: 01 00        	c.nop
   180d0: 00 00 00 00  	v.nop
   180d4: 00 00 00 00 00 00    	m.nop
   180da: 31 c3 8d 58  	l.st.b r13, [r24, 0x58c]
   180de: 01 00        	c.nop
   180e0: 00 00 00 00  	v.nop
   180e4: 00 00 00 00 00 00    	m.nop
   180ea: 31 c5 cd 58  	l.st.w r13, [r24, 0x594]
   180ee: 01 00        	c.nop
   180f0: 00 00 00 00  	v.nop
   180f4: 00 00 00 00 00 00    	m.nop
   180fa: 31 c4 cd 58  	l.st.w r13, [r24, 0x590]
   180fe: 01 00        	c.nop
   18100: 00 00 00 00  	v.nop
   18104: 00 00 00 00 00 00    	m.nop
   1810a: 31 c7 cd 58  	l.st.w r13, [r24, 0x59c]
   1810e: 01 00        	c.nop
   18110: 00 00 00 00  	v.nop
   18114: 00 00 00 00 00 00    	m.nop
   1811a: 31 c6 cd 58  	l.st.w r13, [r24, 0x598]
   1811e: 01 00        	c.nop
   18120: 00 00 00 00  	v.nop
   18124: 00 00 00 00 00 00    	m.nop
   1812a: 31 c1 cd 5a  	l.st.w r13, [r24, 0x5a4]
   1812e: 01 00        	c.nop
   18130: 00 00 00 00  	v.nop
   18134: 00 00 00 00 00 00    	m.nop
   1813a: 31 c0 cd 5a  	l.st.w r13, [r24, 0x5a0]
   1813e: 01 00        	c.nop
   18140: 00 00 00 00  	v.nop
   18144: 00 00 00 00 00 00    	m.nop
   1814a: 31 c3 cd 5a  	l.st.w r13, [r24, 0x5ac]
   1814e: 01 00        	c.nop
   18150: 00 00 00 00  	v.nop
   18154: 00 00 00 00 00 00    	m.nop
   1815a: 31 c2 cd 5a  	l.st.w r13, [r24, 0x5a8]
   1815e: 01 00        	c.nop
   18160: 00 00 00 00  	v.nop
   18164: 00 00 00 00 00 00    	m.nop
   1816a: b1 c4 8d 5a  	l.st.b r13, [r24, 0x5b2]
   1816e: 01 00        	c.nop
   18170: 00 00 00 00  	v.nop
   18174: 00 00 00 00 00 00    	m.nop
   1817a: bb d9 99 c0  	l.blt.s r27, r25, 0x6 <0x181d0>
   1817e: 01 00        	c.nop
   18180: 00 00 00 00  	v.nop
   18184: 00 00 00 00 00 00    	m.nop
   1818a: e3 46 99 02  	l.srl     r27, r8, r25
   1818e: 01 00        	c.nop
   18190: 00 00 00 00  	v.nop
   18194: 00 00 00 00 00 00    	m.nop
   1819a: a5 de 1a 02  	l.orix r26, r27, 0x1a
   1819e: 01 00        	c.nop
   181a0: 00 00 00 00  	v.nop
   181a4: 00 00 00 00 00 00    	m.nop
   181aa: 31 e0 da 00  	l.st.w r26, [r28]
   181ae: 01 00        	c.nop
   181b0: 00 00 00 00  	v.nop
   181b4: 00 00 00 00 00 00    	m.nop
   181ba: ef e7 07 21  	l.ldi16.s r28, 0xff
   181be: 01 00        	c.nop
   181c0: 00 00 00 00  	v.nop
   181c4: 00 00 00 00 00 00    	m.nop
   181ca: 2b 68 80 01  	l.jal r0, 0xd <0x18290>
   181ce: 01 00        	c.nop
   181d0: 00 00 00 00  	v.nop
   181d4: 00 00 00 00 00 00    	m.nop
   181da: b1 e6 44 00  	l.ld.w r26, [r28, 0x4]
   181de: 01 00        	c.nop
   181e0: 00 00 00 00  	v.nop
   181e4: 00 00 00 00 00 00    	m.nop
   181ea: 65 d6 9d 02  	l.xorix r25, r26, 0x1d
   181ee: 01 00        	c.nop
   181f0: 00 00 00 00  	v.nop
   181f4: 00 00 00 00 00 00    	m.nop
   181fa: 69 ce 00 12  	l.bc.lz.u32 r25, r25
   181fe: 01 00        	c.nop
   18200: 00 00 00 00  	v.nop
   18204: 00 00 00 00 00 00    	m.nop
   1820a: bb d8 99 80  	l.bge.s r27, r25, 0x2 <0x18220>
   1820e: 01 00        	c.nop
   18210: 00 00 00 00  	v.nop
   18214: 00 00 00 00 00 00    	m.nop
   1821a: 2b 68 85 05  	l.jal r0, 0x105d <0x287e0>
   1821e: 01 00        	c.nop
   18220: 00 00 00 00  	v.nop
   18224: 00 00 00 00 00 00    	m.nop
   1822a: 71 11 44 02  	l.ld.w r5, [sp, 0x24]
   1822e: 01 00        	c.nop
   18230: 00 00 00 00  	v.nop
   18234: 00 00 00 00 00 00    	m.nop
   1823a: 2f 40 80 30  	l.ldi_hi.m1 r8, 0x8000
   1823e: 01 00        	c.nop
   18240: 00 00 00 00  	v.nop
   18244: 00 00 00 00 00 00    	m.nop
   1824a: e3 46 99 02  	l.srl     r27, r8, r25
   1824e: 01 00        	c.nop
   18250: 00 00 00 00  	v.nop
   18254: 00 00 00 00 00 00    	m.nop
   1825a: a5 de 1a 02  	l.orix r26, r27, 0x1a
   1825e: 01 00        	c.nop
   18260: 00 00 00 00  	v.nop
   18264: 00 00 00 00 00 00    	m.nop
   1826a: 61 ce 20 00  	l.addi.u r25, r25, 0x20
   1826e: 01 00        	c.nop
   18270: 00 00 00 00  	v.nop
   18274: 00 00 00 00 00 00    	m.nop
   1827a: ef e7 07 21  	l.ldi16.s r28, 0xff
   1827e: 01 00        	c.nop
   18280: 00 00 00 00  	v.nop
   18284: 00 00 00 00 00 00    	m.nop
   1828a: 31 28 da 00  	l.st.w r26, [r5]
   1828e: 01 00        	c.nop
   18290: 00 00 00 00  	v.nop
   18294: 00 00 00 00 00 00    	m.nop
   1829a: b3 06 80 00  	l.movs r26, sip_id
   1829e: 7d 44        	<unknown>
   182a0: 00 00 00 00  	v.nop
   182a4: 00 00 00 00 00 00    	m.nop
   182aa: a5 d6 88 01  	l.andix r26, r26, 0x8
   182ae: 81 46        	<unknown>
   182b0: 00 00 00 00  	v.nop
   182b4: 00 00 00 00 00 00    	m.nop
   182ba: 31 c3 da 72  	l.st.w r26, [r24, 0x72c]
   182be: 7d 54        	<unknown>
   182c0: 00 00 00 00  	v.nop
   182c4: 00 00 00 00 00 00    	m.nop
   182ca: a3 ce 09 18  	<unknown>
   182ce: 89 42        	<unknown>
   182d0: 00 00 00 00  	v.nop
   182d4: 00 00 00 00 00 00    	m.nop
   182da: 31 c2 c5 72  	l.st.w r5, [r24, 0x728]
   182de: 01 00        	c.nop
   182e0: 00 00 00 00  	v.nop
   182e4: 00 00 00 00 00 00    	m.nop
   182ea: 31 c4 cd 72  	l.st.w r13, [r24, 0x730]
   182ee: 01 00        	c.nop
   182f0: 00 00 00 00  	v.nop
   182f4: 00 00 00 00 00 00    	m.nop
   182fa: 31 c5 d9 72  	l.st.w r25, [r24, 0x734]
   182fe: 01 00        	c.nop
   18300: 00 00 00 00  	v.nop
   18304: 00 00 00 00 00 00    	m.nop
   1830a: 31 c7 cd 72  	l.st.w r13, [r24, 0x73c]
   1830e: 01 00        	c.nop
   18310: 00 00 00 00  	v.nop
   18314: 00 00 00 00 00 00    	m.nop
   1831a: 31 c0 ad 74  	l.st.h r13, [r24, 0x740]
   1831e: 01 00        	c.nop
   18320: 00 00 00 00  	v.nop
   18324: 00 00 00 00 00 00    	m.nop
   1832a: 31 c1 c8 74  	l.st.w r8, [r24, 0x744]
   1832e: 01 00        	c.nop
   18330: 00 00 00 00  	v.nop
   18334: 00 00 00 00 00 00    	m.nop
   1833a: 31 c2 c8 74  	l.st.w r8, [r24, 0x748]
   1833e: 01 00        	c.nop
   18340: 00 00 00 00  	v.nop
   18344: 00 00 00 00 00 00    	m.nop
   1834a: 31 c3 c8 74  	l.st.w r8, [r24, 0x74c]
   1834e: 01 00        	c.nop
   18350: 00 00 00 00  	v.nop
   18354: 00 00 00 00 00 00    	m.nop
   1835a: 31 c4 cd 74  	l.st.w r13, [r24, 0x750]
   1835e: 01 00        	c.nop
   18360: 00 00 00 00  	v.nop
   18364: 00 00 00 00 00 00    	m.nop
   1836a: 31 c5 cd 74  	l.st.w r13, [r24, 0x754]
   1836e: 01 00        	c.nop
   18370: 00 00 00 00  	v.nop
   18374: 00 00 00 00 00 00    	m.nop
   1837a: 31 c6 8d 74  	l.st.b r13, [r24, 0x758]
   1837e: 01 00        	c.nop
   18380: 00 00 00 00  	v.nop
   18384: 00 00 00 00 00 00    	m.nop
   1838a: 31 c7 c8 74  	l.st.w r8, [r24, 0x75c]
   1838e: 01 00        	c.nop
   18390: 00 00 00 00  	v.nop
   18394: 00 00 00 00 00 00    	m.nop
   1839a: 31 c0 c8 76  	l.st.w r8, [r24, 0x760]
   1839e: 01 00        	c.nop
   183a0: 00 00 00 00  	v.nop
   183a4: 00 00 00 00 00 00    	m.nop
   183aa: 31 c1 c8 76  	l.st.w r8, [r24, 0x764]
   183ae: 01 00        	c.nop
   183b0: 00 00 00 00  	v.nop
   183b4: 00 00 00 00 00 00    	m.nop
   183ba: 31 c2 cd 76  	l.st.w r13, [r24, 0x768]
   183be: 01 00        	c.nop
   183c0: 00 00 00 00  	v.nop
   183c4: 00 00 00 00 00 00    	m.nop
   183ca: 31 c3 cd 76  	l.st.w r13, [r24, 0x76c]
   183ce: 01 00        	c.nop
   183d0: 00 00 00 00  	v.nop
   183d4: 00 00 00 00 00 00    	m.nop
   183da: 71 c1 8d 78  	l.st.b r13, [r24, 0x785]
   183de: 01 00        	c.nop
   183e0: 00 00 00 00  	v.nop
   183e4: 00 00 00 00 00 00    	m.nop
   183ea: 31 c4 cd 76  	l.st.w r13, [r24, 0x770]
   183ee: 01 00        	c.nop
   183f0: 00 00 00 00  	v.nop
   183f4: 00 00 00 00 00 00    	m.nop
   183fa: 31 c5 cd 76  	l.st.w r13, [r24, 0x774]
   183fe: 01 00        	c.nop
   18400: 00 00 00 00  	v.nop
   18404: 00 00 00 00 00 00    	m.nop
   1840a: 31 c6 8d 76  	l.st.b r13, [r24, 0x778]
   1840e: 01 00        	c.nop
   18410: 00 00 00 00  	v.nop
   18414: 00 00 00 00 00 00    	m.nop
   1841a: 31 c7 cd 76  	l.st.w r13, [r24, 0x77c]
   1841e: 01 00        	c.nop
   18420: 00 00 00 00  	v.nop
   18424: 00 00 00 00 00 00    	m.nop
   1842a: 31 c0 cd 78  	l.st.w r13, [r24, 0x780]
   1842e: 01 00        	c.nop
   18430: 00 00 00 00  	v.nop
   18434: 00 00 00 00 00 00    	m.nop
   1843a: 31 c1 8d 78  	l.st.b r13, [r24, 0x784]
   1843e: 01 00        	c.nop
   18440: 00 00 00 00  	v.nop
   18444: 00 00 00 00 00 00    	m.nop
   1844a: 31 c3 cd 78  	l.st.w r13, [r24, 0x78c]
   1844e: 01 00        	c.nop
   18450: 00 00 00 00  	v.nop
   18454: 00 00 00 00 00 00    	m.nop
   1845a: 31 c2 cd 78  	l.st.w r13, [r24, 0x788]
   1845e: 01 00        	c.nop
   18460: 00 00 00 00  	v.nop
   18464: 00 00 00 00 00 00    	m.nop
   1846a: 31 c5 cd 78  	l.st.w r13, [r24, 0x794]
   1846e: 01 00        	c.nop
   18470: 00 00 00 00  	v.nop
   18474: 00 00 00 00 00 00    	m.nop
   1847a: 31 c4 cd 78  	l.st.w r13, [r24, 0x790]
   1847e: 01 00        	c.nop
   18480: 00 00 00 00  	v.nop
   18484: 00 00 00 00 00 00    	m.nop
   1848a: 31 c7 cd 78  	l.st.w r13, [r24, 0x79c]
   1848e: 01 00        	c.nop
   18490: 00 00 00 00  	v.nop
   18494: 00 00 00 00 00 00    	m.nop
   1849a: 31 c6 cd 78  	l.st.w r13, [r24, 0x798]
   1849e: 01 00        	c.nop
   184a0: 00 00 00 00  	v.nop
   184a4: 00 00 00 00 00 00    	m.nop
   184aa: 31 c1 cd 7a  	l.st.w r13, [r24, 0x7a4]
   184ae: 01 00        	c.nop
   184b0: 00 00 00 00  	v.nop
   184b4: 00 00 00 00 00 00    	m.nop
   184ba: 31 c0 cd 7a  	l.st.w r13, [r24, 0x7a0]
   184be: 01 00        	c.nop
   184c0: 00 00 00 00  	v.nop
   184c4: 00 00 00 00 00 00    	m.nop
   184ca: b1 c2 8d 7a  	l.st.b r13, [r24, 0x7aa]
   184ce: 01 00        	c.nop
   184d0: 00 00 00 00  	v.nop
   184d4: 00 00 00 00 00 00    	m.nop
   184da: 31 c2 ad 7a  	l.st.h r13, [r24, 0x7a8]
   184de: 01 00        	c.nop
   184e0: 00 00 00 00  	v.nop
   184e4: 00 00 00 00 00 00    	m.nop
   184ea: 31 c0 d9 72  	l.st.w r25, [r24, 0x720]
   184ee: 01 00        	c.nop
   184f0: 00 00 00 00  	v.nop
   184f4: 00 00 00 00 00 00    	m.nop
   184fa: 31 50 da 00  	l.st.w r26, [r10]
   184fe: 01 00        	c.nop
   18500: 00 00 00 00  	v.nop
   18504: 00 00 00 00 00 00    	m.nop
   1850a: 73 fe 80 08  	<unknown>
   1850e: 01 00        	c.nop
   18510: 00 00 00 00  	v.nop
   18514: 00 00 00 00 00 00    	m.nop
   1851a: 73 fe 80 08  	<unknown>
   1851e: 01 00        	c.nop
   18520: 00 00 00 00  	v.nop
   18524: 00 00 00 00 00 00    	m.nop
   1852a: 71 c6 0c 58  	l.ld.b r25, [r24, 0x58c]
   1852e: 01 00        	c.nop
   18530: 00 00 00 00  	v.nop
   18534: 00 00 00 00 00 00    	m.nop
   1853a: 31 c2 c5 72  	l.st.w r5, [r24, 0x728]
   1853e: 01 00        	c.nop
   18540: 00 00 00 00  	v.nop
   18544: 00 00 00 00 00 00    	m.nop
   1854a: 65 ce 9c 01  	l.andix r25, r25, 0x1c
   1854e: 01 00        	c.nop
   18550: 00 00 00 00  	v.nop
   18554: 00 00 00 00 00 00    	m.nop
   1855a: 3b ce 00 00  	<unknown>
   1855e: 01 00        	c.nop
   18560: 00 00 00 00  	v.nop
   18564: 00 00 00 00 00 00    	m.nop
   1856a: 71 c6 09 54  	l.ld.b r25, [r24, 0x549]
   1856e: 01 00        	c.nop
   18570: 00 00 00 00  	v.nop
   18574: 00 00 00 00 00 00    	m.nop
   1857a: 65 ce 9c 01  	l.andix r25, r25, 0x1c
   1857e: 01 00        	c.nop
   18580: 00 00 00 00  	v.nop
   18584: 00 00 00 00 00 00    	m.nop
   1858a: bb c9 00 00  	<unknown>
   1858e: 01 00        	c.nop
   18590: 00 00 00 00  	v.nop
   18594: 00 00 00 00 00 00    	m.nop
   1859a: 00 00        	s.nop16
   1859c: 00 00        	s.nop16
   1859e: 1f f0        	<unknown>
   185a0: 00 00 00 00  	v.nop
   185a4: 00 00 00 00 00 00    	m.nop
   185aa: 3b 41 00 00  	<unknown>
   185ae: 01 00        	c.nop
   185b0: 00 00 00 00  	v.nop
   185b4: 00 00 00 00 00 00    	m.nop
   185ba: 31 c2 09 54  	l.ld.b r8, [r24, 0x549]
   185be: 01 00        	c.nop
   185c0: 00 00 00 00  	v.nop
   185c4: 00 00 00 00 00 00    	m.nop
   185ca: 25 42 9c 01  	l.andix r8, r8, 0x1c
   185ce: 01 00        	c.nop
   185d0: 00 00 00 00  	v.nop
   185d4: 00 00 00 00 00 00    	m.nop
   185da: fb 43 40 40  	<unknown>
   185de: 01 00        	c.nop
   185e0: 00 00 00 00  	v.nop
   185e4: 00 00 00 00 00 00    	m.nop
   185ea: 31 c2 08 54  	l.ld.b r8, [r24, 0x548]
   185ee: 01 00        	c.nop
   185f0: 00 00 00 00  	v.nop
   185f4: 00 00 00 00 00 00    	m.nop
   185fa: b1 c6 58 52  	l.ld.w r26, [r24, 0x538]
   185fe: 01 00        	c.nop
   18600: 00 00 00 00  	v.nop
   18604: 00 00 00 00 00 00    	m.nop
   1860a: 71 c6 50 52  	l.ld.w r25, [r24, 0x530]
   1860e: 01 00        	c.nop
   18610: 00 00 00 00  	v.nop
   18614: 00 00 00 00 00 00    	m.nop
   1861a: 25 42 9c 01  	l.andix r8, r8, 0x1c
   1861e: 01 00        	c.nop
   18620: 00 00 00 00  	v.nop
   18624: 00 00 00 00 00 00    	m.nop
   1862a: e3 d6 09 03  	l.slli r27, r26, 0x9
   1862e: 01 00        	c.nop
   18630: 00 00 00 00  	v.nop
   18634: 00 00 00 00 00 00    	m.nop
   1863a: bb 43 00 00  	<unknown>
   1863e: 01 00        	c.nop
   18640: 00 00 00 00  	v.nop
   18644: 00 00 00 00 00 00    	m.nop
   1864a: 00 00        	s.nop16
   1864c: 00 00        	s.nop16
   1864e: 01 44        	<unknown>
   18650: 00 00 00 00  	v.nop
   18654: 00 00 00 00 00 00    	m.nop
   1865a: 3b ce 08 00  	<unknown>
   1865e: 01 00        	c.nop
   18660: 00 00 00 00  	v.nop
   18664: 00 00 00 00 00 00    	m.nop
   1866a: 2f 28 08 15  	l.ldi16.u r5, 0xa100
   1866e: 01 00        	c.nop
   18670: 00 00 00 00  	v.nop
   18674: 00 00 00 00 00 00    	m.nop
   1867a: 2f 28 80 08  	l.ldi_hi r5, 0x4000
   1867e: 01 00        	c.nop
   18680: 00 00 00 00  	v.nop
   18684: 00 00 00 00 00 00    	m.nop
   1868a: a1 2e 00 2f  	l.addi.s r26, r5, -0x100
   1868e: 01 00        	c.nop
   18690: 00 00 00 00  	v.nop
   18694: 00 00 00 00 00 00    	m.nop
   1869a: 00 00        	s.nop16
   1869c: 00 00        	s.nop16
   1869e: 3e 8e        	<unknown>
   186a0: 00 00 00 00  	v.nop
   186a4: 00 00 00 00 00 00    	m.nop
   186aa: 00 00        	s.nop16
   186ac: 00 00        	s.nop16
   186ae: 01 44        	<unknown>
   186b0: 00 00 00 00  	v.nop
   186b4: 00 00 00 00 00 00    	m.nop
   186ba: 3b ce 08 40  	<unknown>
   186be: 01 00        	c.nop
   186c0: 00 00 00 00  	v.nop
   186c4: 00 00 00 00 00 00    	m.nop
   186ca: 2b 40 81 01  	l.jal r0, 0x18 <0x18840>
   186ce: 01 00        	c.nop
   186d0: 00 00 00 00  	v.nop
   186d4: 00 00 00 00 00 00    	m.nop
   186da: b1 c6 50 52  	l.ld.w r26, [r24, 0x530]
   186de: 01 00        	c.nop
   186e0: 00 00 00 00  	v.nop
   186e4: 00 00 00 00 00 00    	m.nop
   186ea: fb d5 40 00  	<unknown>
   186ee: 01 00        	c.nop
   186f0: 00 00 00 00  	v.nop
   186f4: 00 00 00 00 00 00    	m.nop
   186fa: 2f c8 40 29  	l.ldi16.s r25, 0x4800
   186fe: 01 00        	c.nop
   18700: 00 00 00 00  	v.nop
   18704: 00 00 00 00 00 00    	m.nop
   1870a: 2b 48 85 01  	l.jal r0, 0x59 <0x18c90>
   1870e: 01 00        	c.nop
   18710: 00 00 00 00  	v.nop
   18714: 00 00 00 00 00 00    	m.nop
   1871a: 00 00        	s.nop16
   1871c: 00 00        	s.nop16
   1871e: 3e 8d        	<unknown>
   18720: 00 00 00 00  	v.nop
   18724: 00 00 00 00 00 00    	m.nop
   1872a: 00 00        	s.nop16
   1872c: 00 00        	s.nop16
   1872e: 01 44        	<unknown>
   18730: 00 00 00 00  	v.nop
   18734: 00 00 00 00 00 00    	m.nop
   1873a: bb c8 08 00  	<unknown>
   1873e: 01 00        	c.nop
   18740: 00 00 00 00  	v.nop
   18744: 00 00 00 00 00 00    	m.nop
   1874a: 00 00        	s.nop16
   1874c: 00 00        	s.nop16
   1874e: 3a 8d        	<unknown>
   18750: 00 00 00 00  	v.nop
   18754: 00 00 00 00 00 00    	m.nop
   1875a: 23 da 1a 18  	<unknown>
   1875e: 85 42        	<unknown>
   18760: 00 00 00 00  	v.nop
   18764: 00 00 00 00 00 00    	m.nop
   1876a: 31 40 cc 00  	l.st.w r12, [r8]
   1876e: 2e 94        	<unknown>
   18770: 00 00 00 00  	v.nop
   18774: 00 00 00 00 00 00    	m.nop
   1877a: 31 40 c5 00  	l.st.w r5, [r8]
   1877e: 01 00        	c.nop
   18780: 00 00 00 00  	v.nop
   18784: 00 00 00 00 00 00    	m.nop
   1878a: 31 46 c5 fe  	l.st.w r5, [r8, -0x8]
   1878e: 01 00        	c.nop
   18790: 00 00 00 00  	v.nop
   18794: 00 00 00 00 00 00    	m.nop
   1879a: 31 42 c5 00  	l.st.w r5, [r8, 0x8]
   1879e: 01 00        	c.nop
   187a0: 00 00 00 00  	v.nop
   187a4: 00 00 00 00 00 00    	m.nop
   187aa: 31 c2 85 54  	l.st.b r5, [r24, 0x548]
   187ae: 01 00        	c.nop
   187b0: 00 00 00 00  	v.nop
   187b4: 00 00 00 00 00 00    	m.nop
   187ba: 00 00        	s.nop16
   187bc: 00 00        	s.nop16
   187be: 01 44        	<unknown>
   187c0: 00 00 00 00  	v.nop
   187c4: 00 00 00 00 00 00    	m.nop
   187ca: bb ca 68 7f  	<unknown>
   187ce: 01 00        	c.nop
   187d0: 00 00 00 00  	v.nop
   187d4: 00 00 00 00 00 00    	m.nop
   187da: 2f 28 08 29  	l.ldi16.s r5, 0x4100
   187de: 01 00        	c.nop
   187e0: 00 00 00 00  	v.nop
   187e4: 00 00 00 00 00 00    	m.nop
   187ea: af 2f 91 08  	l.ldi_hi r5, 0x423e
   187ee: 01 00        	c.nop
   187f0: 00 00 00 00  	v.nop
   187f4: 00 00 00 00 00 00    	m.nop
   187fa: a1 2e 00 2f  	l.addi.s r26, r5, -0x100
   187fe: 01 00        	c.nop
   18800: 00 00 00 00  	v.nop
   18804: 00 00 00 00 00 00    	m.nop
   1880a: 00 00        	s.nop16
   1880c: 00 00        	s.nop16
   1880e: 3e 8e        	<unknown>
   18810: 00 00 00 00  	v.nop
   18814: 00 00 00 00 00 00    	m.nop
   1881a: 00 00        	s.nop16
   1881c: 00 00        	s.nop16
   1881e: 01 44        	<unknown>
   18820: 00 00 00 00  	v.nop
   18824: 00 00 00 00 00 00    	m.nop
   1882a: bb c8 08 00  	<unknown>
   1882e: 01 00        	c.nop
   18830: 00 00 00 00  	v.nop
   18834: 00 00 00 00 00 00    	m.nop
   1883a: 00 00        	s.nop16
   1883c: 00 00        	s.nop16
   1883e: 3a 8e        	<unknown>
   18840: 00 00 00 00  	v.nop
   18844: 00 00 00 00 00 00    	m.nop
   1884a: 31 c2 0d 58  	l.ld.b r8, [r24, 0x58d]
   1884e: 01 00        	c.nop
   18850: 00 00 00 00  	v.nop
   18854: 00 00 00 00 00 00    	m.nop
   1885a: ef 2f 07 21  	l.ldi16.s r5, 0xff
   1885e: 01 00        	c.nop
   18860: 00 00 00 00  	v.nop
   18864: 00 00 00 00 00 00    	m.nop
   1886a: 63 de 1c 18  	<unknown>
   1886e: 01 00        	c.nop
   18870: 00 00 00 00  	v.nop
   18874: 00 00 00 00 00 00    	m.nop
   1887a: 25 42 85 01  	l.andix r8, r8, 0x5
   1887e: 01 00        	c.nop
   18880: 00 00 00 00  	v.nop
   18884: 00 00 00 00 00 00    	m.nop
   1888a: 7b 43 00 00  	<unknown>
   1888e: 01 00        	c.nop
   18890: 00 00 00 00  	v.nop
   18894: 00 00 00 00 00 00    	m.nop
   1889a: 31 c2 5c 52  	l.ld.w r8, [r24, 0x53c]
   1889e: 01 00        	c.nop
   188a0: 00 00 00 00  	v.nop
   188a4: 00 00 00 00 00 00    	m.nop
   188aa: f1 c6 40 54  	l.ld.w r27, [r24, 0x540]
   188ae: 01 00        	c.nop
   188b0: 00 00 00 00  	v.nop
   188b4: 00 00 00 00 00 00    	m.nop
   188ba: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   188be: 01 00        	c.nop
   188c0: 00 00 00 00  	v.nop
   188c4: 00 00 00 00 00 00    	m.nop
   188ca: 21 2f 01 2f  	l.addi.s r28, r5, -0xff
   188ce: 01 00        	c.nop
   188d0: 00 00 00 00  	v.nop
   188d4: 00 00 00 00 00 00    	m.nop
   188da: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   188de: 01 00        	c.nop
   188e0: 00 00 00 00  	v.nop
   188e4: 00 00 00 00 00 00    	m.nop
   188ea: 6f 28 80 00  	l.ldi_hi r5, 0x1
   188ee: 22 04        	<unknown>
   188f0: 00 00 00 00  	v.nop
   188f4: 00 00 00 00 00 00    	m.nop
   188fa: 25 42 9c 01  	l.andix r8, r8, 0x1c
   188fe: 01 00        	c.nop
   18900: 00 00 00 00  	v.nop
   18904: 00 00 00 00 00 00    	m.nop
   1890a: 23 df 18 03  	l.slli r28, r27, 0x18
   1890e: 01 00        	c.nop
   18910: 00 00 00 00  	v.nop
   18914: 00 00 00 00 00 00    	m.nop
   1891a: 61 2f 03 00  	l.addi.u r29, r5, 0x3
   1891e: 01 00        	c.nop
   18920: 00 00 00 00  	v.nop
   18924: 00 00 00 00 00 00    	m.nop
   1892a: 25 e2 08 02  	l.orix r8, r28, 0x8
   1892e: 01 00        	c.nop
   18930: 00 00 00 00  	v.nop
   18934: 00 00 00 00 00 00    	m.nop
   1893a: 25 47 1d 02  	l.orix r28, r8, 0x1d
   1893e: 01 00        	c.nop
   18940: 00 00 00 00  	v.nop
   18944: 00 00 00 00 00 00    	m.nop
   1894a: 2b 38 80 01  	l.jal r0, 0x7 <0x189b0>
   1894e: 01 00        	c.nop
   18950: 00 00 00 00  	v.nop
   18954: 00 00 00 00 00 00    	m.nop
   1895a: f1 c6 40 54  	l.ld.w r27, [r24, 0x540]
   1895e: 01 00        	c.nop
   18960: 00 00 00 00  	v.nop
   18964: 00 00 00 00 00 00    	m.nop
   1896a: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   1896e: 01 00        	c.nop
   18970: 00 00 00 00  	v.nop
   18974: 00 00 00 00 00 00    	m.nop
   1897a: 6f 28 80 00  	l.ldi_hi r5, 0x1
   1897e: 01 00        	c.nop
   18980: 00 00 00 00  	v.nop
   18984: 00 00 00 00 00 00    	m.nop
   1898a: 23 da 18 03  	l.slli r8, r27, 0x18
   1898e: 01 00        	c.nop
   18990: 00 00 00 00  	v.nop
   18994: 00 00 00 00 00 00    	m.nop
   1899a: 25 2a 08 02  	l.orix r8, r5, 0x8
   1899e: 01 00        	c.nop
   189a0: 00 00 00 00  	v.nop
   189a4: 00 00 00 00 00 00    	m.nop
   189aa: 21 47 02 00  	l.addi.u r28, r8, 0x2
   189ae: 01 00        	c.nop
   189b0: 00 00 00 00  	v.nop
   189b4: 00 00 00 00 00 00    	m.nop
   189ba: 25 da 93 01  	l.andix r8, r27, 0x13
   189be: 52 9d        	<unknown>
   189c0: 00 00 00 00  	v.nop
   189c4: 00 00 00 00 00 00    	m.nop
   189ca: 31 c8 dc 00  	l.st.w r28, [r25]
   189ce: 45 8c        	<unknown>
   189d0: 00 00 00 00  	v.nop
   189d4: 00 00 00 00 00 00    	m.nop
   189da: 31 d0 c8 00  	l.st.w r8, [r26]
   189de: be 8d        	<unknown>
   189e0: 00 00 00 00  	v.nop
   189e4: 00 00 00 00 00 00    	m.nop
   189ea: 33 fa 80 00  	<unknown>
   189ee: 01 00        	c.nop
   189f0: 00 00 00 00  	v.nop
   189f4: 00 00 00 00 00 00    	m.nop
   189fa: 33 fa 80 00  	<unknown>
   189fe: 01 00        	c.nop
   18a00: 00 00 00 00  	v.nop
   18a04: 00 00 00 00 00 00    	m.nop
   18a0a: b1 c6 58 52  	l.ld.w r26, [r24, 0x538]
   18a0e: 81 46        	<unknown>
   18a10: 00 00 00 00  	v.nop
   18a14: 00 00 00 00 00 00    	m.nop
   18a1a: 31 c2 50 52  	l.ld.w r8, [r24, 0x530]
   18a1e: 01 00        	c.nop
   18a20: 00 00 00 00  	v.nop
   18a24: 00 00 00 00 00 00    	m.nop
   18a2a: 71 c6 44 56  	l.ld.w r25, [r24, 0x564]
   18a2e: 01 00        	c.nop
   18a30: 00 00 00 00  	v.nop
   18a34: 00 00 00 00 00 00    	m.nop
   18a3a: 00 00        	s.nop16
   18a3c: 00 00        	s.nop16
   18a3e: 26 0d        	<unknown>
   18a40: 00 00 00 00  	v.nop
   18a44: 00 00 00 00 00 00    	m.nop
   18a4a: bb 40 0d 00  	<unknown>
   18a4e: 01 00        	c.nop
   18a50: 00 00 00 00  	v.nop
   18a54: 00 00 00 00 00 00    	m.nop
   18a5a: 00 00        	s.nop16
   18a5c: 00 00        	s.nop16
   18a5e: ba 8d        	<unknown>
   18a60: 00 00 00 00  	v.nop
   18a64: 00 00 00 00 00 00    	m.nop
   18a6a: 23 d2 1b 18  	<unknown>
   18a6e: 01 00        	c.nop
   18a70: 00 00 00 00  	v.nop
   18a74: 00 00 00 00 00 00    	m.nop
   18a7a: a3 ce 19 03  	l.slli r26, r25, 0x19
   18a7e: 8e 0c        	<unknown>
   18a80: 00 00 00 00  	v.nop
   18a84: 00 00 00 00 00 00    	m.nop
   18a8a: ef e7 07 21  	l.ldi16.s r28, 0xff
   18a8e: 2e 94        	<unknown>
   18a90: 00 00 00 00  	v.nop
   18a94: 00 00 00 00 00 00    	m.nop
   18a9a: 00 00        	s.nop16
   18a9c: 00 00        	s.nop16
   18a9e: d6 9c        	<unknown>
   18aa0: 00 00 00 00  	v.nop
   18aa4: 00 00 00 00 00 00    	m.nop
   18aaa: 31 40 d9 00  	l.st.w r25, [r8]
   18aae: 05 0d        	<unknown>
   18ab0: 00 00 00 00  	v.nop
   18ab4: 00 00 00 00 00 00    	m.nop
   18aba: 31 41 da 00  	l.st.w r26, [r8, 0x4]
   18abe: 01 00        	c.nop
   18ac0: 00 00 00 00  	v.nop
   18ac4: 00 00 00 00 00 00    	m.nop
   18aca: 31 c2 48 72  	l.ld.w r8, [r24, 0x728]
   18ace: 85 42        	<unknown>
   18ad0: 00 00 00 00  	v.nop
   18ad4: 00 00 00 00 00 00    	m.nop
   18ada: 3b 28 28 fe  	l.blt.u r5, r8, -0xe0 <0x17cd0>
   18ade: 01 00        	c.nop
   18ae0: 00 00 00 00  	v.nop
   18ae4: 00 00 00 00 00 00    	m.nop
   18aea: 31 c2 01 74  	l.ld.b r8, [r24, 0x741]
   18aee: 01 00        	c.nop
   18af0: 00 00 00 00  	v.nop
   18af4: 00 00 00 00 00 00    	m.nop
   18afa: 25 42 9c 01  	l.andix r8, r8, 0x1c
   18afe: 01 00        	c.nop
   18b00: 00 00 00 00  	v.nop
   18b04: 00 00 00 00 00 00    	m.nop
   18b0a: bb 41 00 00  	<unknown>
   18b0e: 01 00        	c.nop
   18b10: 00 00 00 00  	v.nop
   18b14: 00 00 00 00 00 00    	m.nop
   18b1a: 00 00        	s.nop16
   18b1c: 00 00        	s.nop16
   18b1e: 1f f0        	<unknown>
   18b20: 00 00 00 00  	v.nop
   18b24: 00 00 00 00 00 00    	m.nop
   18b2a: 3b 41 00 00  	<unknown>
   18b2e: 01 00        	c.nop
   18b30: 00 00 00 00  	v.nop
   18b34: 00 00 00 00 00 00    	m.nop
   18b3a: 31 c2 01 74  	l.ld.b r8, [r24, 0x741]
   18b3e: 01 00        	c.nop
   18b40: 00 00 00 00  	v.nop
   18b44: 00 00 00 00 00 00    	m.nop
   18b4a: 25 42 9c 01  	l.andix r8, r8, 0x1c
   18b4e: 01 00        	c.nop
   18b50: 00 00 00 00  	v.nop
   18b54: 00 00 00 00 00 00    	m.nop
   18b5a: 3b 46 00 7e  	<unknown>
   18b5e: 01 00        	c.nop
   18b60: 00 00 00 00  	v.nop
   18b64: 00 00 00 00 00 00    	m.nop
   18b6a: 31 c2 00 74  	l.ld.b r8, [r24, 0x740]
   18b6e: 01 00        	c.nop
   18b70: 00 00 00 00  	v.nop
   18b74: 00 00 00 00 00 00    	m.nop
   18b7a: b1 c6 50 72  	l.ld.w r26, [r24, 0x730]
   18b7e: 01 00        	c.nop
   18b80: 00 00 00 00  	v.nop
   18b84: 00 00 00 00 00 00    	m.nop
   18b8a: 71 c6 48 72  	l.ld.w r25, [r24, 0x728]
   18b8e: 01 00        	c.nop
   18b90: 00 00 00 00  	v.nop
   18b94: 00 00 00 00 00 00    	m.nop
   18b9a: 25 42 9c 01  	l.andix r8, r8, 0x1c
   18b9e: 01 00        	c.nop
   18ba0: 00 00 00 00  	v.nop
   18ba4: 00 00 00 00 00 00    	m.nop
   18baa: 00 00        	s.nop16
   18bac: 00 00        	s.nop16
   18bae: 26 0d        	<unknown>
   18bb0: 00 00 00 00  	v.nop
   18bb4: 00 00 00 00 00 00    	m.nop
   18bba: bb 41 20 00  	<unknown>
   18bbe: 01 00        	c.nop
   18bc0: 00 00 00 00  	v.nop
   18bc4: 00 00 00 00 00 00    	m.nop
   18bca: 00 00        	s.nop16
   18bcc: 00 00        	s.nop16
   18bce: 01 44        	<unknown>
   18bd0: 00 00 00 00  	v.nop
   18bd4: 00 00 00 00 00 00    	m.nop
   18bda: 3b cc 28 00  	<unknown>
   18bde: 01 00        	c.nop
   18be0: 00 00 00 00  	v.nop
   18be4: 00 00 00 00 00 00    	m.nop
   18bea: 2f 20 08 15  	l.ldi16.u r4, 0xa100
   18bee: 01 00        	c.nop
   18bf0: 00 00 00 00  	v.nop
   18bf4: 00 00 00 00 00 00    	m.nop
   18bfa: 2f 20 80 08  	l.ldi_hi r4, 0x4000
   18bfe: 01 00        	c.nop
   18c00: 00 00 00 00  	v.nop
   18c04: 00 00 00 00 00 00    	m.nop
   18c0a: e1 26 00 2f  	l.addi.s r27, r4, -0x100
   18c0e: 01 00        	c.nop
   18c10: 00 00 00 00  	v.nop
   18c14: 00 00 00 00 00 00    	m.nop
   18c1a: 00 00        	s.nop16
   18c1c: 00 00        	s.nop16
   18c1e: 3e 8e        	<unknown>
   18c20: 00 00 00 00  	v.nop
   18c24: 00 00 00 00 00 00    	m.nop
   18c2a: 00 00        	s.nop16
   18c2c: 00 00        	s.nop16
   18c2e: 01 44        	<unknown>
   18c30: 00 00 00 00  	v.nop
   18c34: 00 00 00 00 00 00    	m.nop
   18c3a: 3b cc 28 40  	<unknown>
   18c3e: 01 00        	c.nop
   18c40: 00 00 00 00  	v.nop
   18c44: 00 00 00 00 00 00    	m.nop
   18c4a: 2b 00 83 01  	l.jal r0, 0x30 <0x18f40>
   18c4e: 01 00        	c.nop
   18c50: 00 00 00 00  	v.nop
   18c54: 00 00 00 00 00 00    	m.nop
   18c5a: 33 01 80 00  	l.movs r4, sip_id
   18c5e: 01 00        	c.nop
   18c60: 00 00 00 00  	v.nop
   18c64: 00 00 00 00 00 00    	m.nop
   18c6a: b1 c6 50 52  	l.ld.w r26, [r24, 0x530]
   18c6e: 46 02        	<unknown>
   18c70: 00 00 00 00  	v.nop
   18c74: 00 00 00 00 00 00    	m.nop
   18c7a: 25 21 91 01  	l.andix r4, r4, 0x11
   18c7e: 01 00        	c.nop
   18c80: 00 00 00 00  	v.nop
   18c84: 00 00 00 00 00 00    	m.nop
   18c8a: 65 26 10 02  	l.orix r25, r4, 0x10
   18c8e: 01 00        	c.nop
   18c90: 00 00 00 00  	v.nop
   18c94: 00 00 00 00 00 00    	m.nop
   18c9a: 31 c1 08 54  	l.ld.b r4, [r24, 0x548]
   18c9e: 01 00        	c.nop
   18ca0: 00 00 00 00  	v.nop
   18ca4: 00 00 00 00 00 00    	m.nop
   18caa: f1 c6 58 52  	l.ld.w r27, [r24, 0x538]
   18cae: 01 00        	c.nop
   18cb0: 00 00 00 00  	v.nop
   18cb4: 00 00 00 00 00 00    	m.nop
   18cba: 25 21 9c 01  	l.andix r4, r4, 0x1c
   18cbe: 01 00        	c.nop
   18cc0: 00 00 00 00  	v.nop
   18cc4: 00 00 00 00 00 00    	m.nop
   18cca: 00 00        	s.nop16
   18ccc: 00 00        	s.nop16
   18cce: a6 0d        	<unknown>
   18cd0: 00 00 00 00  	v.nop
   18cd4: 00 00 00 00 00 00    	m.nop
   18cda: 7b 21 00 00  	<unknown>
   18cde: 01 00        	c.nop
   18ce0: 00 00 00 00  	v.nop
   18ce4: 00 00 00 00 00 00    	m.nop
   18cea: 00 00        	s.nop16
   18cec: 00 00        	s.nop16
   18cee: 3e 8e        	<unknown>
   18cf0: 00 00 00 00  	v.nop
   18cf4: 00 00 00 00 00 00    	m.nop
   18cfa: 00 00        	s.nop16
   18cfc: 00 00        	s.nop16
   18cfe: 01 44        	<unknown>
   18d00: 00 00 00 00  	v.nop
   18d04: 00 00 00 00 00 00    	m.nop
   18d0a: bb d5 68 3d  	<unknown>
   18d0e: 01 00        	c.nop
   18d10: 00 00 00 00  	v.nop
   18d14: 00 00 00 00 00 00    	m.nop
   18d1a: 2b 70 80 01  	l.jal r0, 0xe <0x18df0>
   18d1e: 01 00        	c.nop
   18d20: 00 00 00 00  	v.nop
   18d24: 00 00 00 00 00 00    	m.nop
   18d2a: 00 00        	s.nop16
   18d2c: 00 00        	s.nop16
   18d2e: 3e 8e        	<unknown>
   18d30: 00 00 00 00  	v.nop
   18d34: 00 00 00 00 00 00    	m.nop
   18d3a: 00 00        	s.nop16
   18d3c: 00 00        	s.nop16
   18d3e: 01 44        	<unknown>
   18d40: 00 00 00 00  	v.nop
   18d44: 00 00 00 00 00 00    	m.nop
   18d4a: bb d0 08 00  	<unknown>
   18d4e: 01 00        	c.nop
   18d50: 00 00 00 00  	v.nop
   18d54: 00 00 00 00 00 00    	m.nop
   18d5a: 00 00        	s.nop16
   18d5c: 00 00        	s.nop16
   18d5e: 3a 8e        	<unknown>
   18d60: 00 00 00 00  	v.nop
   18d64: 00 00 00 00 00 00    	m.nop
   18d6a: 23 d9 1c 18  	<unknown>
   18d6e: 85 42        	<unknown>
   18d70: 00 00 00 00  	v.nop
   18d74: 00 00 00 00 00 00    	m.nop
   18d7a: 31 20 cc 00  	l.st.w r12, [r4]
   18d7e: 2e 92        	<unknown>
   18d80: 00 00 00 00  	v.nop
   18d84: 00 00 00 00 00 00    	m.nop
   18d8a: 31 20 c5 00  	l.st.w r5, [r4]
   18d8e: 01 00        	c.nop
   18d90: 00 00 00 00  	v.nop
   18d94: 00 00 00 00 00 00    	m.nop
   18d9a: 31 26 c5 fe  	l.st.w r5, [r4, -0x8]
   18d9e: 01 00        	c.nop
   18da0: 00 00 00 00  	v.nop
   18da4: 00 00 00 00 00 00    	m.nop
   18daa: 31 22 c5 00  	l.st.w r5, [r4, 0x8]
   18dae: 01 00        	c.nop
   18db0: 00 00 00 00  	v.nop
   18db4: 00 00 00 00 00 00    	m.nop
   18dba: 31 c2 85 54  	l.st.b r5, [r24, 0x548]
   18dbe: 01 00        	c.nop
   18dc0: 00 00 00 00  	v.nop
   18dc4: 00 00 00 00 00 00    	m.nop
   18dca: 00 00        	s.nop16
   18dcc: 00 00        	s.nop16
   18dce: 3e 8e        	<unknown>
   18dd0: 00 00 00 00  	v.nop
   18dd4: 00 00 00 00 00 00    	m.nop
   18dda: 00 00        	s.nop16
   18ddc: 00 00        	s.nop16
   18dde: 01 44        	<unknown>
   18de0: 00 00 00 00  	v.nop
   18de4: 00 00 00 00 00 00    	m.nop
   18dea: 3b d2 68 3d  	<unknown>
   18dee: 01 00        	c.nop
   18df0: 00 00 00 00  	v.nop
   18df4: 00 00 00 00 00 00    	m.nop
   18dfa: 00 00        	s.nop16
   18dfc: 00 00        	s.nop16
   18dfe: 3a 8e        	<unknown>
   18e00: 00 00 00 00  	v.nop
   18e04: 00 00 00 00 00 00    	m.nop
   18e0a: 2b 30 ee 3f  	l.jal r0, -0x11a <0x17c60>
   18e0e: 01 00        	c.nop
   18e10: 00 00 00 00  	v.nop
   18e14: 00 00 00 00 00 00    	m.nop
   18e1a: 00 00        	s.nop16
   18e1c: 00 00        	s.nop16
   18e1e: be 8d        	<unknown>
   18e20: 00 00 00 00  	v.nop
   18e24: 00 00 00 00 00 00    	m.nop
   18e2a: 00 00        	s.nop16
   18e2c: 00 00        	s.nop16
   18e2e: 01 44        	<unknown>
   18e30: 00 00 00 00  	v.nop
   18e34: 00 00 00 00 00 00    	m.nop
   18e3a: bb c8 08 00  	<unknown>
   18e3e: 01 00        	c.nop
   18e40: 00 00 00 00  	v.nop
   18e44: 00 00 00 00 00 00    	m.nop
   18e4a: 00 00        	s.nop16
   18e4c: 00 00        	s.nop16
   18e4e: ba 8d        	<unknown>
   18e50: 00 00 00 00  	v.nop
   18e54: 00 00 00 00 00 00    	m.nop
   18e5a: 23 d2 1b 18  	<unknown>
   18e5e: 05 42        	<unknown>
   18e60: 00 00 00 00  	v.nop
   18e64: 00 00 00 00 00 00    	m.nop
   18e6a: 31 40 cc 00  	l.st.w r12, [r8]
   18e6e: 2e 94        	<unknown>
   18e70: 00 00 00 00  	v.nop
   18e74: 00 00 00 00 00 00    	m.nop
   18e7a: 31 40 c4 00  	l.st.w r4, [r8]
   18e7e: 01 00        	c.nop
   18e80: 00 00 00 00  	v.nop
   18e84: 00 00 00 00 00 00    	m.nop
   18e8a: 31 46 c4 fe  	l.st.w r4, [r8, -0x8]
   18e8e: 01 00        	c.nop
   18e90: 00 00 00 00  	v.nop
   18e94: 00 00 00 00 00 00    	m.nop
   18e9a: 31 42 c4 00  	l.st.w r4, [r8, 0x8]
   18e9e: 01 00        	c.nop
   18ea0: 00 00 00 00  	v.nop
   18ea4: 00 00 00 00 00 00    	m.nop
   18eaa: 31 c0 84 74  	l.st.b r4, [r24, 0x740]
   18eae: 01 00        	c.nop
   18eb0: 00 00 00 00  	v.nop
   18eb4: 00 00 00 00 00 00    	m.nop
   18eba: 00 00        	s.nop16
   18ebc: 00 00        	s.nop16
   18ebe: 01 44        	<unknown>
   18ec0: 00 00 00 00  	v.nop
   18ec4: 00 00 00 00 00 00    	m.nop
   18eca: bb cc 48 7f  	<unknown>
   18ece: 01 00        	c.nop
   18ed0: 00 00 00 00  	v.nop
   18ed4: 00 00 00 00 00 00    	m.nop
   18eda: 2f 20 08 29  	l.ldi16.s r4, 0x4100
   18ede: 01 00        	c.nop
   18ee0: 00 00 00 00  	v.nop
   18ee4: 00 00 00 00 00 00    	m.nop
   18eea: af 27 91 08  	l.ldi_hi r4, 0x423e
   18eee: 01 00        	c.nop
   18ef0: 00 00 00 00  	v.nop
   18ef4: 00 00 00 00 00 00    	m.nop
   18efa: e1 26 00 2f  	l.addi.s r27, r4, -0x100
   18efe: 01 00        	c.nop
   18f00: 00 00 00 00  	v.nop
   18f04: 00 00 00 00 00 00    	m.nop
   18f0a: 00 00        	s.nop16
   18f0c: 00 00        	s.nop16
   18f0e: 3e 8e        	<unknown>
   18f10: 00 00 00 00  	v.nop
   18f14: 00 00 00 00 00 00    	m.nop
   18f1a: 00 00        	s.nop16
   18f1c: 00 00        	s.nop16
   18f1e: 01 44        	<unknown>
   18f20: 00 00 00 00  	v.nop
   18f24: 00 00 00 00 00 00    	m.nop
   18f2a: bb c8 08 00  	<unknown>
   18f2e: 01 00        	c.nop
   18f30: 00 00 00 00  	v.nop
   18f34: 00 00 00 00 00 00    	m.nop
   18f3a: 00 00        	s.nop16
   18f3c: 00 00        	s.nop16
   18f3e: 3a 8e        	<unknown>
   18f40: 00 00 00 00  	v.nop
   18f44: 00 00 00 00 00 00    	m.nop
   18f4a: 31 c2 54 72  	l.ld.w r8, [r24, 0x734]
   18f4e: 72 9d        	<unknown>
   18f50: 00 00 00 00  	v.nop
   18f54: 00 00 00 00 00 00    	m.nop
   18f5a: ef 27 7f 1f  	l.ldi16.u r4, 0xffff
   18f5e: d2 9d        	<unknown>
   18f60: 00 00 00 00  	v.nop
   18f64: 00 00 00 00 00 00    	m.nop
   18f6a: 21 27 01 2f  	l.addi.s r28, r4, -0xff
   18f6e: 85 42        	<unknown>
   18f70: 00 00 00 00  	v.nop
   18f74: 00 00 00 00 00 00    	m.nop
   18f7a: 61 27 03 00  	l.addi.u r29, r4, 0x3
   18f7e: be 8c        	<unknown>
   18f80: 00 00 00 00  	v.nop
   18f84: 00 00 00 00 00 00    	m.nop
   18f8a: 23 41 08 03  	l.slli r4, r8, 0x8
   18f8e: 01 00        	c.nop
   18f90: 00 00 00 00  	v.nop
   18f94: 00 00 00 00 00 00    	m.nop
   18f9a: 25 21 9c 01  	l.andix r4, r4, 0x1c
   18f9e: 01 00        	c.nop
   18fa0: 00 00 00 00  	v.nop
   18fa4: 00 00 00 00 00 00    	m.nop
   18faa: 25 42 93 01  	l.andix r8, r8, 0x13
   18fae: 01 00        	c.nop
   18fb0: 00 00 00 00  	v.nop
   18fb4: 00 00 00 00 00 00    	m.nop
   18fba: 25 21 1d 02  	l.orix r4, r4, 0x1d
   18fbe: 45 8c        	<unknown>
   18fc0: 00 00 00 00  	v.nop
   18fc4: 00 00 00 00 00 00    	m.nop
   18fca: 31 d0 c4 00  	l.st.w r4, [r26]
   18fce: 01 00        	c.nop
   18fd0: 00 00 00 00  	v.nop
   18fd4: 00 00 00 00 00 00    	m.nop
   18fda: 31 d8 c8 00  	l.st.w r8, [r27]
   18fde: 01 00        	c.nop
   18fe0: 00 00 00 00  	v.nop
   18fe4: 00 00 00 00 00 00    	m.nop
   18fea: 33 f9 80 00  	<unknown>
   18fee: 01 00        	c.nop
   18ff0: 00 00 00 00  	v.nop
   18ff4: 00 00 00 00 00 00    	m.nop
   18ffa: 33 f9 80 00  	<unknown>
   18ffe: 01 00        	c.nop
   19000: 00 00 00 00  	v.nop
   19004: 00 00 00 00 00 00    	m.nop
   1900a: 31 c2 50 72  	l.ld.w r8, [r24, 0x730]
   1900e: 01 00        	c.nop
   19010: 00 00 00 00  	v.nop
   19014: 00 00 00 00 00 00    	m.nop
   1901a: 31 c1 48 72  	l.ld.w r4, [r24, 0x728]
   1901e: 01 00        	c.nop
   19020: 00 00 00 00  	v.nop
   19024: 00 00 00 00 00 00    	m.nop
   1902a: 71 c1 85 78  	l.st.b r5, [r24, 0x785]
   1902e: 01 00        	c.nop
   19030: 00 00 00 00  	v.nop
   19034: 00 00 00 00 00 00    	m.nop
   1903a: 31 c6 44 74  	l.ld.w r24, [r24, 0x744]
   1903e: 01 00        	c.nop
   19040: 00 00 00 00  	v.nop
   19044: 00 00 00 00 00 00    	m.nop
   1904a: a3 46 09 03  	l.slli r26, r8, 0x9
   1904e: 01 44        	<unknown>
   19050: 00 00 00 00  	v.nop
   19054: 00 00 00 00 00 00    	m.nop
   1905a: bb 20 08 00  	<unknown>
   1905e: 01 00        	c.nop
   19060: 00 00 00 00  	v.nop
   19064: 00 00 00 00 00 00    	m.nop
   1906a: 00 00        	s.nop16
   1906c: 00 00        	s.nop16
   1906e: ba 8c        	<unknown>
   19070: 00 00 00 00  	v.nop
   19074: 00 00 00 00 00 00    	m.nop
   1907a: 23 c2 19 03  	l.slli r8, r24, 0x19
   1907e: 0e 0c        	<unknown>
   19080: 00 00 00 00  	v.nop
   19084: 00 00 00 00 00 00    	m.nop
   1908a: 23 d1 19 18  	<unknown>
   1908e: 56 9c        	<unknown>
   19090: 00 00 00 00  	v.nop
   19094: 00 00 00 00 00 00    	m.nop
   1909a: 00 00        	s.nop16
   1909c: 00 00        	s.nop16
   1909e: 5a 92        	<unknown>
   190a0: 00 00 00 00  	v.nop
   190a4: 00 00 00 00 00 00    	m.nop
   190aa: 31 20 d8 00  	l.st.w r24, [r4]
   190ae: 05 04        	<unknown>
   190b0: 00 00 00 00  	v.nop
   190b4: 00 00 00 00 00 00    	m.nop
   190ba: 31 21 c8 00  	l.st.w r8, [r4, 0x4]
   190be: 01 00        	c.nop
   190c0: 00 00 00 00  	v.nop
   190c4: 00 00 00 00 00 00    	m.nop
   190ca: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   190ce: 01 00        	c.nop
   190d0: 00 00 00 00  	v.nop
   190d4: 00 00 00 00 00 00    	m.nop
   190da: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   190de: 01 00        	c.nop
   190e0: 00 00 00 00  	v.nop
   190e4: 00 00 00 00 00 00    	m.nop
   190ea: 2b 78 eb 3f  	l.jal r0, -0x141 <0x17cd0>
   190ee: 01 00        	c.nop
   190f0: 00 00 00 00  	v.nop
   190f4: 00 00 00 00 00 00    	m.nop
   190fa: 2f b8 00 19  	l.ldi16.u r23, 0xc000
   190fe: 05 4d        	<unknown>
   19100: 00 00 00 00  	v.nop
   19104: 00 00 00 00 00 00    	m.nop
   1910a: 2f c0 29 21  	l.ldi16.s r24, 0x520
   1910e: 01 00        	c.nop
   19110: 00 00 00 00  	v.nop
   19114: 00 00 00 00 00 00    	m.nop
   1911a: 2f b8 80 08  	l.ldi_hi r23, 0x4000
   1911e: 01 00        	c.nop
   19120: 00 00 00 00  	v.nop
   19124: 00 00 00 00 00 00    	m.nop
   1912a: 31 12 de 06  	l.st.w r30, [sp, 0x68]
   1912e: 01 00        	c.nop
   19130: 00 00 00 00  	v.nop
   19134: 00 00 00 00 00 00    	m.nop
   1913a: 2b 50 80 01  	l.jal r0, 0xa <0x191d0>
   1913e: 01 00        	c.nop
   19140: 00 00 00 00  	v.nop
   19144: 00 00 00 00 00 00    	m.nop
   1914a: 31 c9 54 72  	l.ld.w r4, [r25, 0x734]
   1914e: 01 00        	c.nop
   19150: 00 00 00 00  	v.nop
   19154: 00 00 00 00 00 00    	m.nop
   1915a: 23 e2 1d 18  	<unknown>
   1915e: 01 00        	c.nop
   19160: 00 00 00 00  	v.nop
   19164: 00 00 00 00 00 00    	m.nop
   1916a: ef e7 07 21  	l.ldi16.s r28, 0xff
   1916e: 4a 94        	<unknown>
   19170: 00 00 00 00  	v.nop
   19174: 00 00 00 00 00 00    	m.nop
   1917a: 31 40 c4 00  	l.st.w r4, [r8]
   1917e: 01 00        	c.nop
   19180: 00 00 00 00  	v.nop
   19184: 00 00 00 00 00 00    	m.nop
   1918a: 31 47 da fe  	l.st.w r26, [r8, -0x4]
   1918e: 05 4d        	<unknown>
   19190: 00 00 00 00  	v.nop
   19194: 00 00 00 00 00 00    	m.nop
   1919a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   1919e: 01 00        	c.nop
   191a0: 00 00 00 00  	v.nop
   191a4: 00 00 00 00 00 00    	m.nop
   191aa: 21 c6 70 2d  	l.addi.s r24, r24, -0x290
   191ae: 01 00        	c.nop
   191b0: 00 00 00 00  	v.nop
   191b4: 00 00 00 00 00 00    	m.nop
   191ba: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   191be: 01 00        	c.nop
   191c0: 00 00 00 00  	v.nop
   191c4: 00 00 00 00 00 00    	m.nop
   191ca: fb c0 40 02  	<unknown>
   191ce: 01 00        	c.nop
   191d0: 00 00 00 00  	v.nop
   191d4: 00 00 00 00 00 00    	m.nop
   191da: 61 11 60 23  	l.addi.s r5, sp, 0x360
   191de: 81 46        	<unknown>
   191e0: 00 00 00 00  	v.nop
   191e4: 00 00 00 00 00 00    	m.nop
   191ea: 63 2e 98 18  	<unknown>
   191ee: 7d 5f        	<unknown>
   191f0: 00 00 00 00  	v.nop
   191f4: 00 00 00 00 00 00    	m.nop
   191fa: 31 c9 cd 52  	l.st.w r13, [r25, 0x524]
   191fe: 12 8e        	<unknown>
   19200: 00 00 00 00  	v.nop
   19204: 00 00 00 00 00 00    	m.nop
   1920a: f1 26 40 00  	l.ld.w r27, [r4]
   1920e: 01 00        	c.nop
   19210: 00 00 00 00  	v.nop
   19214: 00 00 00 00 00 00    	m.nop
   1921a: 31 16 c5 12  	l.st.w r5, [sp, 0x138]
   1921e: ea 82        	<unknown>
   19220: 00 00 00 00  	v.nop
   19224: 00 00 00 00 00 00    	m.nop
   1922a: 31 ca 9a 52  	l.st.b r26, [r25, 0x528]
   1922e: 01 00        	c.nop
   19230: 00 00 00 00  	v.nop
   19234: 00 00 00 00 00 00    	m.nop
   1923a: b1 ca 8d 52  	l.st.b r13, [r25, 0x52a]
   1923e: 01 00        	c.nop
   19240: 00 00 00 00  	v.nop
   19244: 00 00 00 00 00 00    	m.nop
   1924a: 25 d9 9e 02  	l.xorix r4, r27, 0x1e
   1924e: 01 00        	c.nop
   19250: 00 00 00 00  	v.nop
   19254: 00 00 00 00 00 00    	m.nop
   1925a: a9 26 00 12  	l.bc.lz.u32 r26, r4
   1925e: 01 00        	c.nop
   19260: 00 00 00 00  	v.nop
   19264: 00 00 00 00 00 00    	m.nop
   1926a: 2f 20 01 21  	l.ldi16.s r4, 0x20
   1926e: 01 00        	c.nop
   19270: 00 00 00 00  	v.nop
   19274: 00 00 00 00 00 00    	m.nop
   1927a: bb d0 84 c0  	l.blt.s r26, r4, 0x2 <0x19290>
   1927e: 01 00        	c.nop
   19280: 00 00 00 00  	v.nop
   19284: 00 00 00 00 00 00    	m.nop
   1928a: 2b 30 f5 03  	l.jal r0, 0xf56 <0x287e0>
   1928e: 01 00        	c.nop
   19290: 00 00 00 00  	v.nop
   19294: 00 00 00 00 00 00    	m.nop
   1929a: 2f e8 80 30  	l.ldi_hi.m1 r29, 0x8000
   1929e: 01 00        	c.nop
   192a0: 00 00 00 00  	v.nop
   192a4: 00 00 00 00 00 00    	m.nop
   192aa: 23 e9 9a 02  	l.srl     r4, r29, r26
   192ae: 01 00        	c.nop
   192b0: 00 00 00 00  	v.nop
   192b4: 00 00 00 00 00 00    	m.nop
   192ba: 25 21 1b 02  	l.orix r4, r4, 0x1b
   192be: 01 00        	c.nop
   192c0: 00 00 00 00  	v.nop
   192c4: 00 00 00 00 00 00    	m.nop
   192ca: 31 e0 c4 00  	l.st.w r4, [r28]
   192ce: 01 00        	c.nop
   192d0: 00 00 00 00  	v.nop
   192d4: 00 00 00 00 00 00    	m.nop
   192da: 31 17 4c 04  	l.ld.w r28, [sp, 0x4c]
   192de: 01 00        	c.nop
   192e0: 00 00 00 00  	v.nop
   192e4: 00 00 00 00 00 00    	m.nop
   192ea: 23 d2 01 03  	l.slli r8, r26, 0x1
   192ee: 01 00        	c.nop
   192f0: 00 00 00 00  	v.nop
   192f4: 00 00 00 00 00 00    	m.nop
   192fa: 21 41 01 00  	l.addi.u r4, r8, 0x1
   192fe: 01 00        	c.nop
   19300: 00 00 00 00  	v.nop
   19304: 00 00 00 00 00 00    	m.nop
   1930a: 31 c8 c4 54  	l.st.w r4, [r25, 0x540]
   1930e: 01 00        	c.nop
   19310: 00 00 00 00  	v.nop
   19314: 00 00 00 00 00 00    	m.nop
   1931a: 31 cf cd 5e  	l.st.w r13, [r25, 0x5fc]
   1931e: 01 00        	c.nop
   19320: 00 00 00 00  	v.nop
   19324: 00 00 00 00 00 00    	m.nop
   1932a: 31 ca cd 62  	l.st.w r13, [r25, 0x628]
   1932e: 01 00        	c.nop
   19330: 00 00 00 00  	v.nop
   19334: 00 00 00 00 00 00    	m.nop
   1933a: 31 cd cd 5c  	l.st.w r13, [r25, 0x5d4]
   1933e: 01 00        	c.nop
   19340: 00 00 00 00  	v.nop
   19344: 00 00 00 00 00 00    	m.nop
   1934a: 31 c8 cd 5e  	l.st.w r13, [r25, 0x5e0]
   1934e: 01 00        	c.nop
   19350: 00 00 00 00  	v.nop
   19354: 00 00 00 00 00 00    	m.nop
   1935a: 31 cd cd 52  	l.st.w r13, [r25, 0x534]
   1935e: 01 00        	c.nop
   19360: 00 00 00 00  	v.nop
   19364: 00 00 00 00 00 00    	m.nop
   1936a: 31 ce da 52  	l.st.w r26, [r25, 0x538]
   1936e: 01 00        	c.nop
   19370: 00 00 00 00  	v.nop
   19374: 00 00 00 00 00 00    	m.nop
   1937a: 31 cf c8 52  	l.st.w r8, [r25, 0x53c]
   1937e: 01 00        	c.nop
   19380: 00 00 00 00  	v.nop
   19384: 00 00 00 00 00 00    	m.nop
   1938a: 31 c9 cd 54  	l.st.w r13, [r25, 0x544]
   1938e: 01 00        	c.nop
   19390: 00 00 00 00  	v.nop
   19394: 00 00 00 00 00 00    	m.nop
   1939a: 31 ca ad 54  	l.st.h r13, [r25, 0x548]
   1939e: 01 00        	c.nop
   193a0: 00 00 00 00  	v.nop
   193a4: 00 00 00 00 00 00    	m.nop
   193aa: 31 cb de 54  	l.st.w r30, [r25, 0x54c]
   193ae: 01 00        	c.nop
   193b0: 00 00 00 00  	v.nop
   193b4: 00 00 00 00 00 00    	m.nop
   193ba: 31 cc de 54  	l.st.w r30, [r25, 0x550]
   193be: 01 00        	c.nop
   193c0: 00 00 00 00  	v.nop
   193c4: 00 00 00 00 00 00    	m.nop
   193ca: 31 cd de 54  	l.st.w r30, [r25, 0x554]
   193ce: 01 00        	c.nop
   193d0: 00 00 00 00  	v.nop
   193d4: 00 00 00 00 00 00    	m.nop
   193da: 31 ce cd 54  	l.st.w r13, [r25, 0x558]
   193de: 01 00        	c.nop
   193e0: 00 00 00 00  	v.nop
   193e4: 00 00 00 00 00 00    	m.nop
   193ea: 31 cf cd 54  	l.st.w r13, [r25, 0x55c]
   193ee: 01 00        	c.nop
   193f0: 00 00 00 00  	v.nop
   193f4: 00 00 00 00 00 00    	m.nop
   193fa: 31 c9 de 56  	l.st.w r30, [r25, 0x564]
   193fe: 01 00        	c.nop
   19400: 00 00 00 00  	v.nop
   19404: 00 00 00 00 00 00    	m.nop
   1940a: 31 ca de 56  	l.st.w r30, [r25, 0x568]
   1940e: 01 00        	c.nop
   19410: 00 00 00 00  	v.nop
   19414: 00 00 00 00 00 00    	m.nop
   1941a: 31 cb de 56  	l.st.w r30, [r25, 0x56c]
   1941e: 01 00        	c.nop
   19420: 00 00 00 00  	v.nop
   19424: 00 00 00 00 00 00    	m.nop
   1942a: 31 cc cd 56  	l.st.w r13, [r25, 0x570]
   1942e: 01 00        	c.nop
   19430: 00 00 00 00  	v.nop
   19434: 00 00 00 00 00 00    	m.nop
   1943a: 31 cd cd 56  	l.st.w r13, [r25, 0x574]
   1943e: 01 00        	c.nop
   19440: 00 00 00 00  	v.nop
   19444: 00 00 00 00 00 00    	m.nop
   1944a: 31 ce cd 56  	l.st.w r13, [r25, 0x578]
   1944e: 01 00        	c.nop
   19450: 00 00 00 00  	v.nop
   19454: 00 00 00 00 00 00    	m.nop
   1945a: 31 cf cd 56  	l.st.w r13, [r25, 0x57c]
   1945e: 01 00        	c.nop
   19460: 00 00 00 00  	v.nop
   19464: 00 00 00 00 00 00    	m.nop
   1946a: 31 c9 cd 58  	l.st.w r13, [r25, 0x584]
   1946e: 01 00        	c.nop
   19470: 00 00 00 00  	v.nop
   19474: 00 00 00 00 00 00    	m.nop
   1947a: 31 ca cd 58  	l.st.w r13, [r25, 0x588]
   1947e: 01 00        	c.nop
   19480: 00 00 00 00  	v.nop
   19484: 00 00 00 00 00 00    	m.nop
   1948a: 31 cc ad 5a  	l.st.h r13, [r25, 0x5b0]
   1948e: 01 00        	c.nop
   19490: 00 00 00 00  	v.nop
   19494: 00 00 00 00 00 00    	m.nop
   1949a: 31 c8 da 52  	l.st.w r26, [r25, 0x520]
   1949e: 01 00        	c.nop
   194a0: 00 00 00 00  	v.nop
   194a4: 00 00 00 00 00 00    	m.nop
   194aa: 31 c8 de 72  	l.st.w r30, [r25, 0x720]
   194ae: 01 00        	c.nop
   194b0: 00 00 00 00  	v.nop
   194b4: 00 00 00 00 00 00    	m.nop
   194ba: f1 e6 40 00  	l.ld.w r27, [r28]
   194be: 01 00        	c.nop
   194c0: 00 00 00 00  	v.nop
   194c4: 00 00 00 00 00 00    	m.nop
   194ca: 23 d1 09 03  	l.slli r4, r26, 0x9
   194ce: 01 00        	c.nop
   194d0: 00 00 00 00  	v.nop
   194d4: 00 00 00 00 00 00    	m.nop
   194da: 31 ce 8d 70  	l.st.b r13, [r25, 0x718]
   194de: 5e 92        	<unknown>
   194e0: 00 00 00 00  	v.nop
   194e4: 00 00 00 00 00 00    	m.nop
   194ea: 23 22 9f 0b  	l.srai r8, r4, 0x1f
   194ee: 01 00        	c.nop
   194f0: 00 00 00 00  	v.nop
   194f4: 00 00 00 00 00 00    	m.nop
   194fa: 31 ce c4 5a  	l.st.w r4, [r25, 0x5b8]
   194fe: 01 00        	c.nop
   19500: 00 00 00 00  	v.nop
   19504: 00 00 00 00 00 00    	m.nop
   1950a: 25 d9 9e 02  	l.xorix r4, r27, 0x1e
   1950e: 01 00        	c.nop
   19510: 00 00 00 00  	v.nop
   19514: 00 00 00 00 00 00    	m.nop
   1951a: 31 cf c8 5a  	l.st.w r8, [r25, 0x5bc]
   1951e: 7d 44        	<unknown>
   19520: 00 00 00 00  	v.nop
   19524: 00 00 00 00 00 00    	m.nop
   1952a: a9 26 00 12  	l.bc.lz.u32 r26, r4
   1952e: 01 00        	c.nop
   19530: 00 00 00 00  	v.nop
   19534: 00 00 00 00 00 00    	m.nop
   1953a: 31 ce c5 5c  	l.st.w r5, [r25, 0x5d8]
   1953e: 01 00        	c.nop
   19540: 00 00 00 00  	v.nop
   19544: 00 00 00 00 00 00    	m.nop
   1954a: 31 cf c5 5c  	l.st.w r5, [r25, 0x5dc]
   1954e: 01 00        	c.nop
   19550: 00 00 00 00  	v.nop
   19554: 00 00 00 00 00 00    	m.nop
   1955a: 31 c9 c5 5e  	l.st.w r5, [r25, 0x5e4]
   1955e: 01 00        	c.nop
   19560: 00 00 00 00  	v.nop
   19564: 00 00 00 00 00 00    	m.nop
   1956a: 31 ca c5 5e  	l.st.w r5, [r25, 0x5e8]
   1956e: 01 00        	c.nop
   19570: 00 00 00 00  	v.nop
   19574: 00 00 00 00 00 00    	m.nop
   1957a: 31 cc c5 52  	l.st.w r5, [r25, 0x530]
   1957e: 01 00        	c.nop
   19580: 00 00 00 00  	v.nop
   19584: 00 00 00 00 00 00    	m.nop
   1958a: 31 c8 8d 56  	l.st.b r13, [r25, 0x560]
   1958e: 01 00        	c.nop
   19590: 00 00 00 00  	v.nop
   19594: 00 00 00 00 00 00    	m.nop
   1959a: 71 cb 8d 58  	l.st.b r13, [r25, 0x58d]
   1959e: 01 00        	c.nop
   195a0: 00 00 00 00  	v.nop
   195a4: 00 00 00 00 00 00    	m.nop
   195aa: 31 c8 8d 58  	l.st.b r13, [r25, 0x580]
   195ae: 01 00        	c.nop
   195b0: 00 00 00 00  	v.nop
   195b4: 00 00 00 00 00 00    	m.nop
   195ba: 31 cb 8d 58  	l.st.b r13, [r25, 0x58c]
   195be: 01 00        	c.nop
   195c0: 00 00 00 00  	v.nop
   195c4: 00 00 00 00 00 00    	m.nop
   195ca: 31 cd cd 58  	l.st.w r13, [r25, 0x594]
   195ce: 01 00        	c.nop
   195d0: 00 00 00 00  	v.nop
   195d4: 00 00 00 00 00 00    	m.nop
   195da: 31 cc cd 58  	l.st.w r13, [r25, 0x590]
   195de: 01 00        	c.nop
   195e0: 00 00 00 00  	v.nop
   195e4: 00 00 00 00 00 00    	m.nop
   195ea: 31 cf cd 58  	l.st.w r13, [r25, 0x59c]
   195ee: 01 00        	c.nop
   195f0: 00 00 00 00  	v.nop
   195f4: 00 00 00 00 00 00    	m.nop
   195fa: 31 ce cd 58  	l.st.w r13, [r25, 0x598]
   195fe: 01 00        	c.nop
   19600: 00 00 00 00  	v.nop
   19604: 00 00 00 00 00 00    	m.nop
   1960a: 31 c9 cd 5a  	l.st.w r13, [r25, 0x5a4]
   1960e: 01 00        	c.nop
   19610: 00 00 00 00  	v.nop
   19614: 00 00 00 00 00 00    	m.nop
   1961a: 31 c8 cd 5a  	l.st.w r13, [r25, 0x5a0]
   1961e: 01 00        	c.nop
   19620: 00 00 00 00  	v.nop
   19624: 00 00 00 00 00 00    	m.nop
   1962a: 31 cb cd 5a  	l.st.w r13, [r25, 0x5ac]
   1962e: 01 00        	c.nop
   19630: 00 00 00 00  	v.nop
   19634: 00 00 00 00 00 00    	m.nop
   1963a: 31 ca cd 5a  	l.st.w r13, [r25, 0x5a8]
   1963e: 01 00        	c.nop
   19640: 00 00 00 00  	v.nop
   19644: 00 00 00 00 00 00    	m.nop
   1964a: b1 cc 8d 5a  	l.st.b r13, [r25, 0x5b2]
   1964e: 01 00        	c.nop
   19650: 00 00 00 00  	v.nop
   19654: 00 00 00 00 00 00    	m.nop
   1965a: bb 41 9a c0  	l.blt.s r8, r26, 0x6 <0x196b0>
   1965e: 01 00        	c.nop
   19660: 00 00 00 00  	v.nop
   19664: 00 00 00 00 00 00    	m.nop
   1966a: 23 e9 9a 02  	l.srl     r4, r29, r26
   1966e: 01 00        	c.nop
   19670: 00 00 00 00  	v.nop
   19674: 00 00 00 00 00 00    	m.nop
   1967a: 25 21 1b 02  	l.orix r4, r4, 0x1b
   1967e: 01 00        	c.nop
   19680: 00 00 00 00  	v.nop
   19684: 00 00 00 00 00 00    	m.nop
   1968a: 31 e0 c4 00  	l.st.w r4, [r28]
   1968e: 01 00        	c.nop
   19690: 00 00 00 00  	v.nop
   19694: 00 00 00 00 00 00    	m.nop
   1969a: ef e7 07 21  	l.ldi16.s r28, 0xff
   1969e: 01 00        	c.nop
   196a0: 00 00 00 00  	v.nop
   196a4: 00 00 00 00 00 00    	m.nop
   196aa: 2b 68 80 01  	l.jal r0, 0xd <0x19770>
   196ae: 01 00        	c.nop
   196b0: 00 00 00 00  	v.nop
   196b4: 00 00 00 00 00 00    	m.nop
   196ba: f1 e6 44 00  	l.ld.w r27, [r28, 0x4]
   196be: 01 00        	c.nop
   196c0: 00 00 00 00  	v.nop
   196c4: 00 00 00 00 00 00    	m.nop
   196ca: 25 d9 9e 02  	l.xorix r4, r27, 0x1e
   196ce: 01 00        	c.nop
   196d0: 00 00 00 00  	v.nop
   196d4: 00 00 00 00 00 00    	m.nop
   196da: a9 26 00 12  	l.bc.lz.u32 r26, r4
   196de: 01 00        	c.nop
   196e0: 00 00 00 00  	v.nop
   196e4: 00 00 00 00 00 00    	m.nop
   196ea: bb 40 9a 80  	l.bge.s r8, r26, 0x2 <0x19700>
   196ee: 01 00        	c.nop
   196f0: 00 00 00 00  	v.nop
   196f4: 00 00 00 00 00 00    	m.nop
   196fa: 2b 78 f0 03  	l.jal r0, 0xf0f <0x287e0>
   196fe: 01 00        	c.nop
   19700: 00 00 00 00  	v.nop
   19704: 00 00 00 00 00 00    	m.nop
   1970a: 71 11 44 02  	l.ld.w r5, [sp, 0x24]
   1970e: 01 00        	c.nop
   19710: 00 00 00 00  	v.nop
   19714: 00 00 00 00 00 00    	m.nop
   1971a: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1971e: 01 00        	c.nop
   19720: 00 00 00 00  	v.nop
   19724: 00 00 00 00 00 00    	m.nop
   1972a: 23 21 9a 02  	l.srl     r4, r4, r26
   1972e: 01 00        	c.nop
   19730: 00 00 00 00  	v.nop
   19734: 00 00 00 00 00 00    	m.nop
   1973a: 25 21 1b 02  	l.orix r4, r4, 0x1b
   1973e: 01 00        	c.nop
   19740: 00 00 00 00  	v.nop
   19744: 00 00 00 00 00 00    	m.nop
   1974a: a1 d6 20 00  	l.addi.u r26, r26, 0x20
   1974e: 01 00        	c.nop
   19750: 00 00 00 00  	v.nop
   19754: 00 00 00 00 00 00    	m.nop
   1975a: ef e7 07 21  	l.ldi16.s r28, 0xff
   1975e: 01 00        	c.nop
   19760: 00 00 00 00  	v.nop
   19764: 00 00 00 00 00 00    	m.nop
   1976a: 31 28 c4 00  	l.st.w r4, [r5]
   1976e: 01 00        	c.nop
   19770: 00 00 00 00  	v.nop
   19774: 00 00 00 00 00 00    	m.nop
   1977a: 33 01 80 00  	l.movs r4, sip_id
   1977e: 81 46        	<unknown>
   19780: 00 00 00 00  	v.nop
   19784: 00 00 00 00 00 00    	m.nop
   1978a: 25 21 88 01  	l.andix r4, r4, 0x8
   1978e: 7d 54        	<unknown>
   19790: 00 00 00 00  	v.nop
   19794: 00 00 00 00 00 00    	m.nop
   1979a: 31 cb c4 72  	l.st.w r4, [r25, 0x72c]
   1979e: 89 42        	<unknown>
   197a0: 00 00 00 00  	v.nop
   197a4: 00 00 00 00 00 00    	m.nop
   197aa: 23 d1 09 18  	<unknown>
   197ae: 01 00        	c.nop
   197b0: 00 00 00 00  	v.nop
   197b4: 00 00 00 00 00 00    	m.nop
   197ba: 31 ca c5 72  	l.st.w r5, [r25, 0x728]
   197be: 01 00        	c.nop
   197c0: 00 00 00 00  	v.nop
   197c4: 00 00 00 00 00 00    	m.nop
   197ca: 31 cc cd 72  	l.st.w r13, [r25, 0x730]
   197ce: 01 00        	c.nop
   197d0: 00 00 00 00  	v.nop
   197d4: 00 00 00 00 00 00    	m.nop
   197da: 31 cd da 72  	l.st.w r26, [r25, 0x734]
   197de: 01 00        	c.nop
   197e0: 00 00 00 00  	v.nop
   197e4: 00 00 00 00 00 00    	m.nop
   197ea: 31 cf cd 72  	l.st.w r13, [r25, 0x73c]
   197ee: 01 00        	c.nop
   197f0: 00 00 00 00  	v.nop
   197f4: 00 00 00 00 00 00    	m.nop
   197fa: 31 c8 ad 74  	l.st.h r13, [r25, 0x740]
   197fe: 01 00        	c.nop
   19800: 00 00 00 00  	v.nop
   19804: 00 00 00 00 00 00    	m.nop
   1980a: 31 c9 c8 74  	l.st.w r8, [r25, 0x744]
   1980e: 01 00        	c.nop
   19810: 00 00 00 00  	v.nop
   19814: 00 00 00 00 00 00    	m.nop
   1981a: 31 ca c8 74  	l.st.w r8, [r25, 0x748]
   1981e: 01 00        	c.nop
   19820: 00 00 00 00  	v.nop
   19824: 00 00 00 00 00 00    	m.nop
   1982a: 31 cb c8 74  	l.st.w r8, [r25, 0x74c]
   1982e: 01 00        	c.nop
   19830: 00 00 00 00  	v.nop
   19834: 00 00 00 00 00 00    	m.nop
   1983a: 31 cc cd 74  	l.st.w r13, [r25, 0x750]
   1983e: 01 00        	c.nop
   19840: 00 00 00 00  	v.nop
   19844: 00 00 00 00 00 00    	m.nop
   1984a: 31 cd cd 74  	l.st.w r13, [r25, 0x754]
   1984e: 01 00        	c.nop
   19850: 00 00 00 00  	v.nop
   19854: 00 00 00 00 00 00    	m.nop
   1985a: 31 ce 8d 74  	l.st.b r13, [r25, 0x758]
   1985e: 01 00        	c.nop
   19860: 00 00 00 00  	v.nop
   19864: 00 00 00 00 00 00    	m.nop
   1986a: 31 cf c8 74  	l.st.w r8, [r25, 0x75c]
   1986e: 01 00        	c.nop
   19870: 00 00 00 00  	v.nop
   19874: 00 00 00 00 00 00    	m.nop
   1987a: 31 c8 c8 76  	l.st.w r8, [r25, 0x760]
   1987e: 01 00        	c.nop
   19880: 00 00 00 00  	v.nop
   19884: 00 00 00 00 00 00    	m.nop
   1988a: 31 c9 c8 76  	l.st.w r8, [r25, 0x764]
   1988e: 01 00        	c.nop
   19890: 00 00 00 00  	v.nop
   19894: 00 00 00 00 00 00    	m.nop
   1989a: 31 ca cd 76  	l.st.w r13, [r25, 0x768]
   1989e: 01 00        	c.nop
   198a0: 00 00 00 00  	v.nop
   198a4: 00 00 00 00 00 00    	m.nop
   198aa: 31 cb cd 76  	l.st.w r13, [r25, 0x76c]
   198ae: 01 00        	c.nop
   198b0: 00 00 00 00  	v.nop
   198b4: 00 00 00 00 00 00    	m.nop
   198ba: 71 c9 8d 78  	l.st.b r13, [r25, 0x785]
   198be: 01 00        	c.nop
   198c0: 00 00 00 00  	v.nop
   198c4: 00 00 00 00 00 00    	m.nop
   198ca: 31 cc cd 76  	l.st.w r13, [r25, 0x770]
   198ce: 01 00        	c.nop
   198d0: 00 00 00 00  	v.nop
   198d4: 00 00 00 00 00 00    	m.nop
   198da: 31 cd cd 76  	l.st.w r13, [r25, 0x774]
   198de: 01 00        	c.nop
   198e0: 00 00 00 00  	v.nop
   198e4: 00 00 00 00 00 00    	m.nop
   198ea: 31 ce 8d 76  	l.st.b r13, [r25, 0x778]
   198ee: 01 00        	c.nop
   198f0: 00 00 00 00  	v.nop
   198f4: 00 00 00 00 00 00    	m.nop
   198fa: 31 cf cd 76  	l.st.w r13, [r25, 0x77c]
   198fe: 01 00        	c.nop
   19900: 00 00 00 00  	v.nop
   19904: 00 00 00 00 00 00    	m.nop
   1990a: 31 c8 cd 78  	l.st.w r13, [r25, 0x780]
   1990e: 01 00        	c.nop
   19910: 00 00 00 00  	v.nop
   19914: 00 00 00 00 00 00    	m.nop
   1991a: 31 c9 8d 78  	l.st.b r13, [r25, 0x784]
   1991e: 01 00        	c.nop
   19920: 00 00 00 00  	v.nop
   19924: 00 00 00 00 00 00    	m.nop
   1992a: 31 cb cd 78  	l.st.w r13, [r25, 0x78c]
   1992e: 01 00        	c.nop
   19930: 00 00 00 00  	v.nop
   19934: 00 00 00 00 00 00    	m.nop
   1993a: 31 ca cd 78  	l.st.w r13, [r25, 0x788]
   1993e: 01 00        	c.nop
   19940: 00 00 00 00  	v.nop
   19944: 00 00 00 00 00 00    	m.nop
   1994a: 31 cd cd 78  	l.st.w r13, [r25, 0x794]
   1994e: 01 00        	c.nop
   19950: 00 00 00 00  	v.nop
   19954: 00 00 00 00 00 00    	m.nop
   1995a: 31 cc cd 78  	l.st.w r13, [r25, 0x790]
   1995e: 01 00        	c.nop
   19960: 00 00 00 00  	v.nop
   19964: 00 00 00 00 00 00    	m.nop
   1996a: 31 cf cd 78  	l.st.w r13, [r25, 0x79c]
   1996e: 01 00        	c.nop
   19970: 00 00 00 00  	v.nop
   19974: 00 00 00 00 00 00    	m.nop
   1997a: 31 ce cd 78  	l.st.w r13, [r25, 0x798]
   1997e: 01 00        	c.nop
   19980: 00 00 00 00  	v.nop
   19984: 00 00 00 00 00 00    	m.nop
   1998a: 31 c9 cd 7a  	l.st.w r13, [r25, 0x7a4]
   1998e: 01 00        	c.nop
   19990: 00 00 00 00  	v.nop
   19994: 00 00 00 00 00 00    	m.nop
   1999a: 31 c8 cd 7a  	l.st.w r13, [r25, 0x7a0]
   1999e: 01 00        	c.nop
   199a0: 00 00 00 00  	v.nop
   199a4: 00 00 00 00 00 00    	m.nop
   199aa: b1 ca 8d 7a  	l.st.b r13, [r25, 0x7aa]
   199ae: 01 00        	c.nop
   199b0: 00 00 00 00  	v.nop
   199b4: 00 00 00 00 00 00    	m.nop
   199ba: 31 ca ad 7a  	l.st.h r13, [r25, 0x7a8]
   199be: 01 00        	c.nop
   199c0: 00 00 00 00  	v.nop
   199c4: 00 00 00 00 00 00    	m.nop
   199ca: 31 c8 da 72  	l.st.w r26, [r25, 0x720]
   199ce: 01 00        	c.nop
   199d0: 00 00 00 00  	v.nop
   199d4: 00 00 00 00 00 00    	m.nop
   199da: 31 50 c4 00  	l.st.w r4, [r10]
   199de: 01 00        	c.nop
   199e0: 00 00 00 00  	v.nop
   199e4: 00 00 00 00 00 00    	m.nop
   199ea: 33 f9 80 08  	<unknown>
   199ee: 01 00        	c.nop
   199f0: 00 00 00 00  	v.nop
   199f4: 00 00 00 00 00 00    	m.nop
   199fa: 33 f9 80 08  	<unknown>
   199fe: 01 00        	c.nop
   19a00: 00 00 00 00  	v.nop
   19a04: 00 00 00 00 00 00    	m.nop
   19a0a: 31 c9 0c 58  	l.ld.b r4, [r25, 0x58c]
   19a0e: 01 00        	c.nop
   19a10: 00 00 00 00  	v.nop
   19a14: 00 00 00 00 00 00    	m.nop
   19a1a: 31 ca c5 72  	l.st.w r5, [r25, 0x728]
   19a1e: 01 00        	c.nop
   19a20: 00 00 00 00  	v.nop
   19a24: 00 00 00 00 00 00    	m.nop
   19a2a: 25 21 9c 01  	l.andix r4, r4, 0x1c
   19a2e: 01 00        	c.nop
   19a30: 00 00 00 00  	v.nop
   19a34: 00 00 00 00 00 00    	m.nop
   19a3a: 3b 26 00 00  	<unknown>
   19a3e: 01 00        	c.nop
   19a40: 00 00 00 00  	v.nop
   19a44: 00 00 00 00 00 00    	m.nop
   19a4a: 31 c9 09 54  	l.ld.b r4, [r25, 0x549]
   19a4e: 01 00        	c.nop
   19a50: 00 00 00 00  	v.nop
   19a54: 00 00 00 00 00 00    	m.nop
   19a5a: 25 21 9c 01  	l.andix r4, r4, 0x1c
   19a5e: 01 00        	c.nop
   19a60: 00 00 00 00  	v.nop
   19a64: 00 00 00 00 00 00    	m.nop
   19a6a: bb 21 00 00  	<unknown>
   19a6e: 01 00        	c.nop
   19a70: 00 00 00 00  	v.nop
   19a74: 00 00 00 00 00 00    	m.nop
   19a7a: 00 00        	s.nop16
   19a7c: 00 00        	s.nop16
   19a7e: 1f f0        	<unknown>
   19a80: 00 00 00 00  	v.nop
   19a84: 00 00 00 00 00 00    	m.nop
   19a8a: 3b 41 00 00  	<unknown>
   19a8e: 01 00        	c.nop
   19a90: 00 00 00 00  	v.nop
   19a94: 00 00 00 00 00 00    	m.nop
   19a9a: 31 c9 09 54  	l.ld.b r4, [r25, 0x549]
   19a9e: 01 00        	c.nop
   19aa0: 00 00 00 00  	v.nop
   19aa4: 00 00 00 00 00 00    	m.nop
   19aaa: 25 21 9c 01  	l.andix r4, r4, 0x1c
   19aae: 01 00        	c.nop
   19ab0: 00 00 00 00  	v.nop
   19ab4: 00 00 00 00 00 00    	m.nop
   19aba: 7b 27 40 40  	<unknown>
   19abe: 01 00        	c.nop
   19ac0: 00 00 00 00  	v.nop
   19ac4: 00 00 00 00 00 00    	m.nop
   19aca: 31 c9 08 54  	l.ld.b r4, [r25, 0x548]
   19ace: 01 00        	c.nop
   19ad0: 00 00 00 00  	v.nop
   19ad4: 00 00 00 00 00 00    	m.nop
   19ada: b1 ce 50 52  	l.ld.w r26, [r25, 0x530]
   19ade: 01 00        	c.nop
   19ae0: 00 00 00 00  	v.nop
   19ae4: 00 00 00 00 00 00    	m.nop
   19aea: f1 ce 58 52  	l.ld.w r27, [r25, 0x538]
   19aee: 01 00        	c.nop
   19af0: 00 00 00 00  	v.nop
   19af4: 00 00 00 00 00 00    	m.nop
   19afa: 25 21 9c 01  	l.andix r4, r4, 0x1c
   19afe: 01 00        	c.nop
   19b00: 00 00 00 00  	v.nop
   19b04: 00 00 00 00 00 00    	m.nop
   19b0a: fb 26 00 00  	<unknown>
   19b0e: 01 00        	c.nop
   19b10: 00 00 00 00  	v.nop
   19b14: 00 00 00 00 00 00    	m.nop
   19b1a: 23 df 09 03  	l.slli r28, r27, 0x9
   19b1e: 01 00        	c.nop
   19b20: 00 00 00 00  	v.nop
   19b24: 00 00 00 00 00 00    	m.nop
   19b2a: 00 00        	s.nop16
   19b2c: 00 00        	s.nop16
   19b2e: 01 44        	<unknown>
   19b30: 00 00 00 00  	v.nop
   19b34: 00 00 00 00 00 00    	m.nop
   19b3a: 3b d1 28 00  	<unknown>
   19b3e: 01 00        	c.nop
   19b40: 00 00 00 00  	v.nop
   19b44: 00 00 00 00 00 00    	m.nop
   19b4a: 2f 20 08 15  	l.ldi16.u r4, 0xa100
   19b4e: 01 00        	c.nop
   19b50: 00 00 00 00  	v.nop
   19b54: 00 00 00 00 00 00    	m.nop
   19b5a: 2f 20 80 08  	l.ldi_hi r4, 0x4000
   19b5e: 01 00        	c.nop
   19b60: 00 00 00 00  	v.nop
   19b64: 00 00 00 00 00 00    	m.nop
   19b6a: e1 26 00 2f  	l.addi.s r27, r4, -0x100
   19b6e: 01 00        	c.nop
   19b70: 00 00 00 00  	v.nop
   19b74: 00 00 00 00 00 00    	m.nop
   19b7a: 00 00        	s.nop16
   19b7c: 00 00        	s.nop16
   19b7e: be 8e        	<unknown>
   19b80: 00 00 00 00  	v.nop
   19b84: 00 00 00 00 00 00    	m.nop
   19b8a: 00 00        	s.nop16
   19b8c: 00 00        	s.nop16
   19b8e: 01 44        	<unknown>
   19b90: 00 00 00 00  	v.nop
   19b94: 00 00 00 00 00 00    	m.nop
   19b9a: 3b d1 28 40  	<unknown>
   19b9e: 01 00        	c.nop
   19ba0: 00 00 00 00  	v.nop
   19ba4: 00 00 00 00 00 00    	m.nop
   19baa: 2b 20 82 01  	l.jal r0, 0x24 <0x19de0>
   19bae: 01 00        	c.nop
   19bb0: 00 00 00 00  	v.nop
   19bb4: 00 00 00 00 00 00    	m.nop
   19bba: f1 ce 50 52  	l.ld.w r27, [r25, 0x530]
   19bbe: 1e 8d        	<unknown>
   19bc0: 00 00 00 00  	v.nop
   19bc4: 00 00 00 00 00 00    	m.nop
   19bca: 7b d9 00 40  	<unknown>
   19bce: 01 00        	c.nop
   19bd0: 00 00 00 00  	v.nop
   19bd4: 00 00 00 00 00 00    	m.nop
   19bda: 33 01 80 00  	l.movs r4, sip_id
   19bde: 01 00        	c.nop
   19be0: 00 00 00 00  	v.nop
   19be4: 00 00 00 00 00 00    	m.nop
   19bea: f1 ce 50 52  	l.ld.w r27, [r25, 0x530]
   19bee: 46 02        	<unknown>
   19bf0: 00 00 00 00  	v.nop
   19bf4: 00 00 00 00 00 00    	m.nop
   19bfa: 25 21 91 01  	l.andix r4, r4, 0x11
   19bfe: 01 00        	c.nop
   19c00: 00 00 00 00  	v.nop
   19c04: 00 00 00 00 00 00    	m.nop
   19c0a: a5 26 10 02  	l.orix r26, r4, 0x10
   19c0e: 01 00        	c.nop
   19c10: 00 00 00 00  	v.nop
   19c14: 00 00 00 00 00 00    	m.nop
   19c1a: 31 cf 58 52  	l.ld.w r28, [r25, 0x538]
   19c1e: 01 00        	c.nop
   19c20: 00 00 00 00  	v.nop
   19c24: 00 00 00 00 00 00    	m.nop
   19c2a: 31 c9 08 54  	l.ld.b r4, [r25, 0x548]
   19c2e: 01 00        	c.nop
   19c30: 00 00 00 00  	v.nop
   19c34: 00 00 00 00 00 00    	m.nop
   19c3a: ef 2f 07 21  	l.ldi16.s r5, 0xff
   19c3e: 01 00        	c.nop
   19c40: 00 00 00 00  	v.nop
   19c44: 00 00 00 00 00 00    	m.nop
   19c4a: 00 00        	s.nop16
   19c4c: 00 00        	s.nop16
   19c4e: 26 0e        	<unknown>
   19c50: 00 00 00 00  	v.nop
   19c54: 00 00 00 00 00 00    	m.nop
   19c5a: 25 21 85 01  	l.andix r4, r4, 0x5
   19c5e: 01 00        	c.nop
   19c60: 00 00 00 00  	v.nop
   19c64: 00 00 00 00 00 00    	m.nop
   19c6a: fb 26 40 00  	<unknown>
   19c6e: 01 00        	c.nop
   19c70: 00 00 00 00  	v.nop
   19c74: 00 00 00 00 00 00    	m.nop
   19c7a: 00 00        	s.nop16
   19c7c: 00 00        	s.nop16
   19c7e: be 8e        	<unknown>
   19c80: 00 00 00 00  	v.nop
   19c84: 00 00 00 00 00 00    	m.nop
   19c8a: 00 00        	s.nop16
   19c8c: 00 00        	s.nop16
   19c8e: 01 44        	<unknown>
   19c90: 00 00 00 00  	v.nop
   19c94: 00 00 00 00 00 00    	m.nop
   19c9a: fb da 48 3e  	<unknown>
   19c9e: 01 00        	c.nop
   19ca0: 00 00 00 00  	v.nop
   19ca4: 00 00 00 00 00 00    	m.nop
   19caa: 2b 20 86 01  	l.jal r0, 0x64 <0x1a2e0>
   19cae: 01 00        	c.nop
   19cb0: 00 00 00 00  	v.nop
   19cb4: 00 00 00 00 00 00    	m.nop
   19cba: 23 df 09 03  	l.slli r28, r27, 0x9
   19cbe: be 8d        	<unknown>
   19cc0: 00 00 00 00  	v.nop
   19cc4: 00 00 00 00 00 00    	m.nop
   19cca: 00 00        	s.nop16
   19ccc: 00 00        	s.nop16
   19cce: 01 44        	<unknown>
   19cd0: 00 00 00 00  	v.nop
   19cd4: 00 00 00 00 00 00    	m.nop
   19cda: bb d0 08 00  	<unknown>
   19cde: 01 00        	c.nop
   19ce0: 00 00 00 00  	v.nop
   19ce4: 00 00 00 00 00 00    	m.nop
   19cea: 00 00        	s.nop16
   19cec: 00 00        	s.nop16
   19cee: ba 8d        	<unknown>
   19cf0: 00 00 00 00  	v.nop
   19cf4: 00 00 00 00 00 00    	m.nop
   19cfa: 23 e1 1b 18  	<unknown>
   19cfe: 85 42        	<unknown>
   19d00: 00 00 00 00  	v.nop
   19d04: 00 00 00 00 00 00    	m.nop
   19d0a: 31 20 cc 00  	l.st.w r12, [r4]
   19d0e: 2e 92        	<unknown>
   19d10: 00 00 00 00  	v.nop
   19d14: 00 00 00 00 00 00    	m.nop
   19d1a: 31 20 c5 00  	l.st.w r5, [r4]
   19d1e: 01 00        	c.nop
   19d20: 00 00 00 00  	v.nop
   19d24: 00 00 00 00 00 00    	m.nop
   19d2a: 31 26 c5 fe  	l.st.w r5, [r4, -0x8]
   19d2e: 01 00        	c.nop
   19d30: 00 00 00 00  	v.nop
   19d34: 00 00 00 00 00 00    	m.nop
   19d3a: 31 22 c5 00  	l.st.w r5, [r4, 0x8]
   19d3e: 01 00        	c.nop
   19d40: 00 00 00 00  	v.nop
   19d44: 00 00 00 00 00 00    	m.nop
   19d4a: 31 ca 85 54  	l.st.b r5, [r25, 0x548]
   19d4e: 01 00        	c.nop
   19d50: 00 00 00 00  	v.nop
   19d54: 00 00 00 00 00 00    	m.nop
   19d5a: 00 00        	s.nop16
   19d5c: 00 00        	s.nop16
   19d5e: 01 44        	<unknown>
   19d60: 00 00 00 00  	v.nop
   19d64: 00 00 00 00 00 00    	m.nop
   19d6a: bb d7 48 7f  	<unknown>
   19d6e: 01 00        	c.nop
   19d70: 00 00 00 00  	v.nop
   19d74: 00 00 00 00 00 00    	m.nop
   19d7a: 2f 20 08 29  	l.ldi16.s r4, 0x4100
   19d7e: 01 00        	c.nop
   19d80: 00 00 00 00  	v.nop
   19d84: 00 00 00 00 00 00    	m.nop
   19d8a: af 27 91 08  	l.ldi_hi r4, 0x423e
   19d8e: 01 00        	c.nop
   19d90: 00 00 00 00  	v.nop
   19d94: 00 00 00 00 00 00    	m.nop
   19d9a: e1 26 00 2f  	l.addi.s r27, r4, -0x100
   19d9e: 01 00        	c.nop
   19da0: 00 00 00 00  	v.nop
   19da4: 00 00 00 00 00 00    	m.nop
   19daa: 00 00        	s.nop16
   19dac: 00 00        	s.nop16
   19dae: be 8e        	<unknown>
   19db0: 00 00 00 00  	v.nop
   19db4: 00 00 00 00 00 00    	m.nop
   19dba: 00 00        	s.nop16
   19dbc: 00 00        	s.nop16
   19dbe: 01 44        	<unknown>
   19dc0: 00 00 00 00  	v.nop
   19dc4: 00 00 00 00 00 00    	m.nop
   19dca: bb d0 08 00  	<unknown>
   19dce: 01 00        	c.nop
   19dd0: 00 00 00 00  	v.nop
   19dd4: 00 00 00 00 00 00    	m.nop
   19dda: 00 00        	s.nop16
   19ddc: 00 00        	s.nop16
   19dde: ba 8e        	<unknown>
   19de0: 00 00 00 00  	v.nop
   19de4: 00 00 00 00 00 00    	m.nop
   19dea: 31 c9 0d 58  	l.ld.b r4, [r25, 0x58d]
   19dee: 01 00        	c.nop
   19df0: 00 00 00 00  	v.nop
   19df4: 00 00 00 00 00 00    	m.nop
   19dfa: ef 2f 07 21  	l.ldi16.s r5, 0xff
   19dfe: 01 00        	c.nop
   19e00: 00 00 00 00  	v.nop
   19e04: 00 00 00 00 00 00    	m.nop
   19e0a: a3 e6 1d 18  	<unknown>
   19e0e: 01 00        	c.nop
   19e10: 00 00 00 00  	v.nop
   19e14: 00 00 00 00 00 00    	m.nop
   19e1a: 25 21 85 01  	l.andix r4, r4, 0x5
   19e1e: 01 00        	c.nop
   19e20: 00 00 00 00  	v.nop
   19e24: 00 00 00 00 00 00    	m.nop
   19e2a: 7b 23 00 00  	<unknown>
   19e2e: 01 00        	c.nop
   19e30: 00 00 00 00  	v.nop
   19e34: 00 00 00 00 00 00    	m.nop
   19e3a: 31 c9 5c 52  	l.ld.w r4, [r25, 0x53c]
   19e3e: 01 00        	c.nop
   19e40: 00 00 00 00  	v.nop
   19e44: 00 00 00 00 00 00    	m.nop
   19e4a: 31 cf 40 54  	l.ld.w r28, [r25, 0x540]
   19e4e: 01 00        	c.nop
   19e50: 00 00 00 00  	v.nop
   19e54: 00 00 00 00 00 00    	m.nop
   19e5a: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   19e5e: 01 00        	c.nop
   19e60: 00 00 00 00  	v.nop
   19e64: 00 00 00 00 00 00    	m.nop
   19e6a: 21 2a 01 2f  	l.addi.s r8, r5, -0xff
   19e6e: 01 00        	c.nop
   19e70: 00 00 00 00  	v.nop
   19e74: 00 00 00 00 00 00    	m.nop
   19e7a: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   19e7e: 01 00        	c.nop
   19e80: 00 00 00 00  	v.nop
   19e84: 00 00 00 00 00 00    	m.nop
   19e8a: 6f 28 80 00  	l.ldi_hi r5, 0x1
   19e8e: 22 02        	<unknown>
   19e90: 00 00 00 00  	v.nop
   19e94: 00 00 00 00 00 00    	m.nop
   19e9a: 25 21 88 01  	l.andix r4, r4, 0x8
   19e9e: 01 00        	c.nop
   19ea0: 00 00 00 00  	v.nop
   19ea4: 00 00 00 00 00 00    	m.nop
   19eaa: 23 e2 18 03  	l.slli r8, r28, 0x18
   19eae: 01 00        	c.nop
   19eb0: 00 00 00 00  	v.nop
   19eb4: 00 00 00 00 00 00    	m.nop
   19eba: 61 2f 03 00  	l.addi.u r29, r5, 0x3
   19ebe: 01 00        	c.nop
   19ec0: 00 00 00 00  	v.nop
   19ec4: 00 00 00 00 00 00    	m.nop
   19eca: 25 41 04 02  	l.orix r4, r8, 0x4
   19ece: 01 00        	c.nop
   19ed0: 00 00 00 00  	v.nop
   19ed4: 00 00 00 00 00 00    	m.nop
   19eda: 65 27 1d 02  	l.orix r29, r4, 0x1d
   19ede: 01 00        	c.nop
   19ee0: 00 00 00 00  	v.nop
   19ee4: 00 00 00 00 00 00    	m.nop
   19eea: 2b 38 80 01  	l.jal r0, 0x7 <0x19f50>
   19eee: 01 00        	c.nop
   19ef0: 00 00 00 00  	v.nop
   19ef4: 00 00 00 00 00 00    	m.nop
   19efa: 31 cf 40 54  	l.ld.w r28, [r25, 0x540]
   19efe: 01 00        	c.nop
   19f00: 00 00 00 00  	v.nop
   19f04: 00 00 00 00 00 00    	m.nop
   19f0a: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   19f0e: 01 00        	c.nop
   19f10: 00 00 00 00  	v.nop
   19f14: 00 00 00 00 00 00    	m.nop
   19f1a: 6f 28 80 00  	l.ldi_hi r5, 0x1
   19f1e: 01 00        	c.nop
   19f20: 00 00 00 00  	v.nop
   19f24: 00 00 00 00 00 00    	m.nop
   19f2a: 23 e1 18 03  	l.slli r4, r28, 0x18
   19f2e: 01 00        	c.nop
   19f30: 00 00 00 00  	v.nop
   19f34: 00 00 00 00 00 00    	m.nop
   19f3a: 25 29 04 02  	l.orix r4, r5, 0x4
   19f3e: 01 00        	c.nop
   19f40: 00 00 00 00  	v.nop
   19f44: 00 00 00 00 00 00    	m.nop
   19f4a: 61 27 02 00  	l.addi.u r29, r4, 0x2
   19f4e: 01 00        	c.nop
   19f50: 00 00 00 00  	v.nop
   19f54: 00 00 00 00 00 00    	m.nop
   19f5a: 25 e1 93 01  	l.andix r4, r28, 0x13
   19f5e: 3e 8e        	<unknown>
   19f60: 00 00 00 00  	v.nop
   19f64: 00 00 00 00 00 00    	m.nop
   19f6a: 25 21 09 02  	l.orix r4, r4, 0x9
   19f6e: 01 00        	c.nop
   19f70: 00 00 00 00  	v.nop
   19f74: 00 00 00 00 00 00    	m.nop
   19f7a: 23 da 14 10  	<unknown>
   19f7e: 01 00        	c.nop
   19f80: 00 00 00 00  	v.nop
   19f84: 00 00 00 00 00 00    	m.nop
   19f8a: 31 d0 dd 00  	l.st.w r29, [r26]
   19f8e: 01 00        	c.nop
   19f90: 00 00 00 00  	v.nop
   19f94: 00 00 00 00 00 00    	m.nop
   19f9a: 31 40 c4 00  	l.st.w r4, [r8]
   19f9e: 01 00        	c.nop
   19fa0: 00 00 00 00  	v.nop
   19fa4: 00 00 00 00 00 00    	m.nop
   19faa: 33 f9 80 00  	<unknown>
   19fae: 01 00        	c.nop
   19fb0: 00 00 00 00  	v.nop
   19fb4: 00 00 00 00 00 00    	m.nop
   19fba: 33 f9 80 00  	<unknown>
   19fbe: 01 00        	c.nop
   19fc0: 00 00 00 00  	v.nop
   19fc4: 00 00 00 00 00 00    	m.nop
   19fca: 31 ca 58 52  	l.ld.w r8, [r25, 0x538]
   19fce: 01 00        	c.nop
   19fd0: 00 00 00 00  	v.nop
   19fd4: 00 00 00 00 00 00    	m.nop
   19fda: 31 c9 50 52  	l.ld.w r4, [r25, 0x530]
   19fde: 01 00        	c.nop
   19fe0: 00 00 00 00  	v.nop
   19fe4: 00 00 00 00 00 00    	m.nop
   19fea: b1 ce 44 56  	l.ld.w r26, [r25, 0x564]
   19fee: 01 00        	c.nop
   19ff0: 00 00 00 00  	v.nop
   19ff4: 00 00 00 00 00 00    	m.nop
   19ffa: e3 46 09 03  	l.slli r27, r8, 0x9
   19ffe: 01 44        	<unknown>
   1a000: 00 00 00 00  	v.nop
   1a004: 00 00 00 00 00 00    	m.nop
   1a00a: bb 20 08 00  	<unknown>
   1a00e: 01 00        	c.nop
   1a010: 00 00 00 00  	v.nop
   1a014: 00 00 00 00 00 00    	m.nop
   1a01a: 00 00        	s.nop16
   1a01c: 00 00        	s.nop16
   1a01e: 3a 8e        	<unknown>
   1a020: 00 00 00 00  	v.nop
   1a024: 00 00 00 00 00 00    	m.nop
   1a02a: 23 d9 1c 18  	<unknown>
   1a02e: 01 00        	c.nop
   1a030: 00 00 00 00  	v.nop
   1a034: 00 00 00 00 00 00    	m.nop
   1a03a: 23 d2 19 03  	l.slli r8, r26, 0x19
   1a03e: 0e 0d        	<unknown>
   1a040: 00 00 00 00  	v.nop
   1a044: 00 00 00 00 00 00    	m.nop
   1a04a: ef e7 07 21  	l.ldi16.s r28, 0xff
   1a04e: 2e 92        	<unknown>
   1a050: 00 00 00 00  	v.nop
   1a054: 00 00 00 00 00 00    	m.nop
   1a05a: 00 00        	s.nop16
   1a05c: 00 00        	s.nop16
   1a05e: 56 9d        	<unknown>
   1a060: 00 00 00 00  	v.nop
   1a064: 00 00 00 00 00 00    	m.nop
   1a06a: 31 20 da 00  	l.st.w r26, [r4]
   1a06e: 05 04        	<unknown>
   1a070: 00 00 00 00  	v.nop
   1a074: 00 00 00 00 00 00    	m.nop
   1a07a: 31 21 c8 00  	l.st.w r8, [r4, 0x4]
   1a07e: 01 00        	c.nop
   1a080: 00 00 00 00  	v.nop
   1a084: 00 00 00 00 00 00    	m.nop
   1a08a: 31 c9 48 72  	l.ld.w r4, [r25, 0x728]
   1a08e: 05 4d        	<unknown>
   1a090: 00 00 00 00  	v.nop
   1a094: 00 00 00 00 00 00    	m.nop
   1a09a: 3b d4 04 fe  	l.blt.u r26, r4, -0xf0 <0x19190>
   1a09e: 01 00        	c.nop
   1a0a0: 00 00 00 00  	v.nop
   1a0a4: 00 00 00 00 00 00    	m.nop
   1a0aa: 31 c9 01 74  	l.ld.b r4, [r25, 0x741]
   1a0ae: 01 00        	c.nop
   1a0b0: 00 00 00 00  	v.nop
   1a0b4: 00 00 00 00 00 00    	m.nop
   1a0ba: 25 21 9c 01  	l.andix r4, r4, 0x1c
   1a0be: 01 00        	c.nop
   1a0c0: 00 00 00 00  	v.nop
   1a0c4: 00 00 00 00 00 00    	m.nop
   1a0ca: bb 21 00 00  	<unknown>
   1a0ce: 01 00        	c.nop
   1a0d0: 00 00 00 00  	v.nop
   1a0d4: 00 00 00 00 00 00    	m.nop
   1a0da: 00 00        	s.nop16
   1a0dc: 00 00        	s.nop16
   1a0de: 1f f0        	<unknown>
   1a0e0: 00 00 00 00  	v.nop
   1a0e4: 00 00 00 00 00 00    	m.nop
   1a0ea: 3b 41 00 00  	<unknown>
   1a0ee: 01 00        	c.nop
   1a0f0: 00 00 00 00  	v.nop
   1a0f4: 00 00 00 00 00 00    	m.nop
   1a0fa: 31 c9 01 74  	l.ld.b r4, [r25, 0x741]
   1a0fe: 01 00        	c.nop
   1a100: 00 00 00 00  	v.nop
   1a104: 00 00 00 00 00 00    	m.nop
   1a10a: 25 21 9c 01  	l.andix r4, r4, 0x1c
   1a10e: 01 00        	c.nop
   1a110: 00 00 00 00  	v.nop
   1a114: 00 00 00 00 00 00    	m.nop
   1a11a: 3b 22 00 7e  	<unknown>
   1a11e: 01 00        	c.nop
   1a120: 00 00 00 00  	v.nop
   1a124: 00 00 00 00 00 00    	m.nop
   1a12a: 31 c9 00 74  	l.ld.b r4, [r25, 0x740]
   1a12e: 01 00        	c.nop
   1a130: 00 00 00 00  	v.nop
   1a134: 00 00 00 00 00 00    	m.nop
   1a13a: f1 ce 50 72  	l.ld.w r27, [r25, 0x730]
   1a13e: 01 00        	c.nop
   1a140: 00 00 00 00  	v.nop
   1a144: 00 00 00 00 00 00    	m.nop
   1a14a: b1 ce 48 72  	l.ld.w r26, [r25, 0x728]
   1a14e: 01 00        	c.nop
   1a150: 00 00 00 00  	v.nop
   1a154: 00 00 00 00 00 00    	m.nop
   1a15a: 25 21 9c 01  	l.andix r4, r4, 0x1c
   1a15e: 01 00        	c.nop
   1a160: 00 00 00 00  	v.nop
   1a164: 00 00 00 00 00 00    	m.nop
   1a16a: 00 00        	s.nop16
   1a16c: 00 00        	s.nop16
   1a16e: a6 0d        	<unknown>
   1a170: 00 00 00 00  	v.nop
   1a174: 00 00 00 00 00 00    	m.nop
   1a17a: 7b 26 00 00  	<unknown>
   1a17e: 01 00        	c.nop
   1a180: 00 00 00 00  	v.nop
   1a184: 00 00 00 00 00 00    	m.nop
   1a18a: 00 00        	s.nop16
   1a18c: 00 00        	s.nop16
   1a18e: 01 44        	<unknown>
   1a190: 00 00 00 00  	v.nop
   1a194: 00 00 00 00 00 00    	m.nop
   1a19a: fb d0 28 00  	<unknown>
   1a19e: 01 00        	c.nop
   1a1a0: 00 00 00 00  	v.nop
   1a1a4: 00 00 00 00 00 00    	m.nop
   1a1aa: 2f 20 08 15  	l.ldi16.u r4, 0xa100
   1a1ae: 01 00        	c.nop
   1a1b0: 00 00 00 00  	v.nop
   1a1b4: 00 00 00 00 00 00    	m.nop
   1a1ba: 2f 20 80 08  	l.ldi_hi r4, 0x4000
   1a1be: 01 00        	c.nop
   1a1c0: 00 00 00 00  	v.nop
   1a1c4: 00 00 00 00 00 00    	m.nop
   1a1ca: 21 27 00 2f  	l.addi.s r28, r4, -0x100
   1a1ce: 01 00        	c.nop
   1a1d0: 00 00 00 00  	v.nop
   1a1d4: 00 00 00 00 00 00    	m.nop
   1a1da: 00 00        	s.nop16
   1a1dc: 00 00        	s.nop16
   1a1de: be 8e        	<unknown>
   1a1e0: 00 00 00 00  	v.nop
   1a1e4: 00 00 00 00 00 00    	m.nop
   1a1ea: 00 00        	s.nop16
   1a1ec: 00 00        	s.nop16
   1a1ee: 01 44        	<unknown>
   1a1f0: 00 00 00 00  	v.nop
   1a1f4: 00 00 00 00 00 00    	m.nop
   1a1fa: fb d0 28 40  	<unknown>
   1a1fe: 01 00        	c.nop
   1a200: 00 00 00 00  	v.nop
   1a204: 00 00 00 00 00 00    	m.nop
   1a20a: 2b 18 82 01  	l.jal r0, 0x23 <0x1a430>
   1a20e: 01 00        	c.nop
   1a210: 00 00 00 00  	v.nop
   1a214: 00 00 00 00 00 00    	m.nop
   1a21a: 00 00        	s.nop16
   1a21c: 00 00        	s.nop16
   1a21e: be 8e        	<unknown>
   1a220: 00 00 00 00  	v.nop
   1a224: 00 00 00 00 00 00    	m.nop
   1a22a: 00 00        	s.nop16
   1a22c: 00 00        	s.nop16
   1a22e: 01 44        	<unknown>
   1a230: 00 00 00 00  	v.nop
   1a234: 00 00 00 00 00 00    	m.nop
   1a23a: bb d8 08 00  	<unknown>
   1a23e: 01 00        	c.nop
   1a240: 00 00 00 00  	v.nop
   1a244: 00 00 00 00 00 00    	m.nop
   1a24a: 00 00        	s.nop16
   1a24c: 00 00        	s.nop16
   1a24e: ba 8e        	<unknown>
   1a250: 00 00 00 00  	v.nop
   1a254: 00 00 00 00 00 00    	m.nop
   1a25a: 23 e1 1d 18  	<unknown>
   1a25e: 85 42        	<unknown>
   1a260: 00 00 00 00  	v.nop
   1a264: 00 00 00 00 00 00    	m.nop
   1a26a: 31 20 cc 00  	l.st.w r12, [r4]
   1a26e: 2e 92        	<unknown>
   1a270: 00 00 00 00  	v.nop
   1a274: 00 00 00 00 00 00    	m.nop
   1a27a: 31 20 c5 00  	l.st.w r5, [r4]
   1a27e: 01 00        	c.nop
   1a280: 00 00 00 00  	v.nop
   1a284: 00 00 00 00 00 00    	m.nop
   1a28a: 31 26 c5 fe  	l.st.w r5, [r4, -0x8]
   1a28e: 01 00        	c.nop
   1a290: 00 00 00 00  	v.nop
   1a294: 00 00 00 00 00 00    	m.nop
   1a29a: 31 22 c5 00  	l.st.w r5, [r4, 0x8]
   1a29e: 01 00        	c.nop
   1a2a0: 00 00 00 00  	v.nop
   1a2a4: 00 00 00 00 00 00    	m.nop
   1a2aa: 31 ca 85 54  	l.st.b r5, [r25, 0x548]
   1a2ae: 01 00        	c.nop
   1a2b0: 00 00 00 00  	v.nop
   1a2b4: 00 00 00 00 00 00    	m.nop
   1a2ba: 00 00        	s.nop16
   1a2bc: 00 00        	s.nop16
   1a2be: be 8e        	<unknown>
   1a2c0: 00 00 00 00  	v.nop
   1a2c4: 00 00 00 00 00 00    	m.nop
   1a2ca: 00 00        	s.nop16
   1a2cc: 00 00        	s.nop16
   1a2ce: 01 44        	<unknown>
   1a2d0: 00 00 00 00  	v.nop
   1a2d4: 00 00 00 00 00 00    	m.nop
   1a2da: fb d9 68 3d  	<unknown>
   1a2de: 01 00        	c.nop
   1a2e0: 00 00 00 00  	v.nop
   1a2e4: 00 00 00 00 00 00    	m.nop
   1a2ea: 00 00        	s.nop16
   1a2ec: 00 00        	s.nop16
   1a2ee: ba 8e        	<unknown>
   1a2f0: 00 00 00 00  	v.nop
   1a2f4: 00 00 00 00 00 00    	m.nop
   1a2fa: 2b 28 ee 3f  	l.jal r0, -0x11b <0x19140>
   1a2fe: 01 00        	c.nop
   1a300: 00 00 00 00  	v.nop
   1a304: 00 00 00 00 00 00    	m.nop
   1a30a: 00 00        	s.nop16
   1a30c: 00 00        	s.nop16
   1a30e: 3e 8e        	<unknown>
   1a310: 00 00 00 00  	v.nop
   1a314: 00 00 00 00 00 00    	m.nop
   1a31a: 00 00        	s.nop16
   1a31c: 00 00        	s.nop16
   1a31e: 01 44        	<unknown>
   1a320: 00 00 00 00  	v.nop
   1a324: 00 00 00 00 00 00    	m.nop
   1a32a: bb d0 08 00  	<unknown>
   1a32e: 01 00        	c.nop
   1a330: 00 00 00 00  	v.nop
   1a334: 00 00 00 00 00 00    	m.nop
   1a33a: 00 00        	s.nop16
   1a33c: 00 00        	s.nop16
   1a33e: 3a 8e        	<unknown>
   1a340: 00 00 00 00  	v.nop
   1a344: 00 00 00 00 00 00    	m.nop
   1a34a: 23 d9 1c 18  	<unknown>
   1a34e: 85 42        	<unknown>
   1a350: 00 00 00 00  	v.nop
   1a354: 00 00 00 00 00 00    	m.nop
   1a35a: 31 20 cc 00  	l.st.w r12, [r4]
   1a35e: 2e 92        	<unknown>
   1a360: 00 00 00 00  	v.nop
   1a364: 00 00 00 00 00 00    	m.nop
   1a36a: 31 20 c5 00  	l.st.w r5, [r4]
   1a36e: 01 00        	c.nop
   1a370: 00 00 00 00  	v.nop
   1a374: 00 00 00 00 00 00    	m.nop
   1a37a: 31 26 c5 fe  	l.st.w r5, [r4, -0x8]
   1a37e: 01 00        	c.nop
   1a380: 00 00 00 00  	v.nop
   1a384: 00 00 00 00 00 00    	m.nop
   1a38a: 31 22 c5 00  	l.st.w r5, [r4, 0x8]
   1a38e: 01 00        	c.nop
   1a390: 00 00 00 00  	v.nop
   1a394: 00 00 00 00 00 00    	m.nop
   1a39a: 31 c8 85 74  	l.st.b r5, [r25, 0x740]
   1a39e: 01 00        	c.nop
   1a3a0: 00 00 00 00  	v.nop
   1a3a4: 00 00 00 00 00 00    	m.nop
   1a3aa: 00 00        	s.nop16
   1a3ac: 00 00        	s.nop16
   1a3ae: 01 44        	<unknown>
   1a3b0: 00 00 00 00  	v.nop
   1a3b4: 00 00 00 00 00 00    	m.nop
   1a3ba: fb d7 48 7f  	<unknown>
   1a3be: 01 00        	c.nop
   1a3c0: 00 00 00 00  	v.nop
   1a3c4: 00 00 00 00 00 00    	m.nop
   1a3ca: 2f 20 08 29  	l.ldi16.s r4, 0x4100
   1a3ce: 01 00        	c.nop
   1a3d0: 00 00 00 00  	v.nop
   1a3d4: 00 00 00 00 00 00    	m.nop
   1a3da: af 27 91 08  	l.ldi_hi r4, 0x423e
   1a3de: 01 00        	c.nop
   1a3e0: 00 00 00 00  	v.nop
   1a3e4: 00 00 00 00 00 00    	m.nop
   1a3ea: 21 27 00 2f  	l.addi.s r28, r4, -0x100
   1a3ee: 01 00        	c.nop
   1a3f0: 00 00 00 00  	v.nop
   1a3f4: 00 00 00 00 00 00    	m.nop
   1a3fa: 00 00        	s.nop16
   1a3fc: 00 00        	s.nop16
   1a3fe: be 8e        	<unknown>
   1a400: 00 00 00 00  	v.nop
   1a404: 00 00 00 00 00 00    	m.nop
   1a40a: 00 00        	s.nop16
   1a40c: 00 00        	s.nop16
   1a40e: 01 44        	<unknown>
   1a410: 00 00 00 00  	v.nop
   1a414: 00 00 00 00 00 00    	m.nop
   1a41a: bb d0 08 00  	<unknown>
   1a41e: 01 00        	c.nop
   1a420: 00 00 00 00  	v.nop
   1a424: 00 00 00 00 00 00    	m.nop
   1a42a: 00 00        	s.nop16
   1a42c: 00 00        	s.nop16
   1a42e: ba 8e        	<unknown>
   1a430: 00 00 00 00  	v.nop
   1a434: 00 00 00 00 00 00    	m.nop
   1a43a: 31 c9 54 72  	l.ld.w r4, [r25, 0x734]
   1a43e: 52 9e        	<unknown>
   1a440: 00 00 00 00  	v.nop
   1a444: 00 00 00 00 00 00    	m.nop
   1a44a: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   1a44e: 3e 8d        	<unknown>
   1a450: 00 00 00 00  	v.nop
   1a454: 00 00 00 00 00 00    	m.nop
   1a45a: 23 da 1d 18  	<unknown>
   1a45e: 01 00        	c.nop
   1a460: 00 00 00 00  	v.nop
   1a464: 00 00 00 00 00 00    	m.nop
   1a46a: e1 2e 01 2f  	l.addi.s r27, r5, -0xff
   1a46e: 01 00        	c.nop
   1a470: 00 00 00 00  	v.nop
   1a474: 00 00 00 00 00 00    	m.nop
   1a47a: 61 2f 03 00  	l.addi.u r29, r5, 0x3
   1a47e: 85 42        	<unknown>
   1a480: 00 00 00 00  	v.nop
   1a484: 00 00 00 00 00 00    	m.nop
   1a48a: a3 27 08 03  	l.slli r30, r4, 0x8
   1a48e: 01 00        	c.nop
   1a490: 00 00 00 00  	v.nop
   1a494: 00 00 00 00 00 00    	m.nop
   1a49a: e5 f6 9b 01  	l.andix r27, r30, 0x1b
   1a49e: 01 00        	c.nop
   1a4a0: 00 00 00 00  	v.nop
   1a4a4: 00 00 00 00 00 00    	m.nop
   1a4aa: 25 21 93 01  	l.andix r4, r4, 0x13
   1a4ae: 01 00        	c.nop
   1a4b0: 00 00 00 00  	v.nop
   1a4b4: 00 00 00 00 00 00    	m.nop
   1a4ba: e5 de 1d 02  	l.orix r27, r27, 0x1d
   1a4be: 01 00        	c.nop
   1a4c0: 00 00 00 00  	v.nop
   1a4c4: 00 00 00 00 00 00    	m.nop
   1a4ca: 25 21 09 02  	l.orix r4, r4, 0x9
   1a4ce: 01 00        	c.nop
   1a4d0: 00 00 00 00  	v.nop
   1a4d4: 00 00 00 00 00 00    	m.nop
   1a4da: 31 40 db 00  	l.st.w r27, [r8]
   1a4de: 01 00        	c.nop
   1a4e0: 00 00 00 00  	v.nop
   1a4e4: 00 00 00 00 00 00    	m.nop
   1a4ea: 31 e0 c4 00  	l.st.w r4, [r28]
   1a4ee: 01 00        	c.nop
   1a4f0: 00 00 00 00  	v.nop
   1a4f4: 00 00 00 00 00 00    	m.nop
   1a4fa: 33 f9 80 00  	<unknown>
   1a4fe: 01 00        	c.nop
   1a500: 00 00 00 00  	v.nop
   1a504: 00 00 00 00 00 00    	m.nop
   1a50a: 33 f9 80 00  	<unknown>
   1a50e: 01 00        	c.nop
   1a510: 00 00 00 00  	v.nop
   1a514: 00 00 00 00 00 00    	m.nop
   1a51a: 31 ca 50 72  	l.ld.w r8, [r25, 0x730]
   1a51e: 01 00        	c.nop
   1a520: 00 00 00 00  	v.nop
   1a524: 00 00 00 00 00 00    	m.nop
   1a52a: 31 c9 48 72  	l.ld.w r4, [r25, 0x728]
   1a52e: 01 00        	c.nop
   1a530: 00 00 00 00  	v.nop
   1a534: 00 00 00 00 00 00    	m.nop
   1a53a: 71 c9 85 78  	l.st.b r5, [r25, 0x785]
   1a53e: 01 00        	c.nop
   1a540: 00 00 00 00  	v.nop
   1a544: 00 00 00 00 00 00    	m.nop
   1a54a: 71 ce 44 74  	l.ld.w r25, [r25, 0x744]
   1a54e: 01 00        	c.nop
   1a550: 00 00 00 00  	v.nop
   1a554: 00 00 00 00 00 00    	m.nop
   1a55a: e3 46 09 03  	l.slli r27, r8, 0x9
   1a55e: 01 44        	<unknown>
   1a560: 00 00 00 00  	v.nop
   1a564: 00 00 00 00 00 00    	m.nop
   1a56a: bb 20 08 00  	<unknown>
   1a56e: 01 00        	c.nop
   1a570: 00 00 00 00  	v.nop
   1a574: 00 00 00 00 00 00    	m.nop
   1a57a: 00 00        	s.nop16
   1a57c: 00 00        	s.nop16
   1a57e: 3a 8d        	<unknown>
   1a580: 00 00 00 00  	v.nop
   1a584: 00 00 00 00 00 00    	m.nop
   1a58a: 23 d9 1a 18  	<unknown>
   1a58e: 05 4d        	<unknown>
   1a590: 00 00 00 00  	v.nop
   1a594: 00 00 00 00 00 00    	m.nop
   1a59a: 23 ca 19 03  	l.slli r8, r25, 0x19
   1a59e: 8e 0c        	<unknown>
   1a5a0: 00 00 00 00  	v.nop
   1a5a4: 00 00 00 00 00 00    	m.nop
   1a5aa: ef e7 07 21  	l.ldi16.s r28, 0xff
   1a5ae: 5a 92        	<unknown>
   1a5b0: 00 00 00 00  	v.nop
   1a5b4: 00 00 00 00 00 00    	m.nop
   1a5ba: 00 00        	s.nop16
   1a5bc: 00 00        	s.nop16
   1a5be: d6 9c        	<unknown>
   1a5c0: 00 00 00 00  	v.nop
   1a5c4: 00 00 00 00 00 00    	m.nop
   1a5ca: 31 20 d9 00  	l.st.w r25, [r4]
   1a5ce: 05 04        	<unknown>
   1a5d0: 00 00 00 00  	v.nop
   1a5d4: 00 00 00 00 00 00    	m.nop
   1a5da: 31 21 c8 00  	l.st.w r8, [r4, 0x4]
   1a5de: 01 00        	c.nop
   1a5e0: 00 00 00 00  	v.nop
   1a5e4: 00 00 00 00 00 00    	m.nop
   1a5ea: 2b 58 eb 3f  	l.jal r0, -0x145 <0x19190>
   1a5ee: 01 00        	c.nop
   1a5f0: 00 00 00 00  	v.nop
   1a5f4: 00 00 00 00 00 00    	m.nop
   1a5fa: b1 23 40 00  	l.ld.w r14, [r4]
   1a5fe: 7d 59        	<unknown>
   1a600: 00 00 00 00  	v.nop
   1a604: 00 00 00 00 00 00    	m.nop
   1a60a: 31 15 5c 06  	l.ld.w r20, [sp, 0x7c]
   1a60e: 12 84        	<unknown>
   1a610: 00 00 00 00  	v.nop
   1a614: 00 00 00 00 00 00    	m.nop
   1a61a: 25 71 92 02  	l.xorix r4, r14, 0x12
   1a61e: 01 00        	c.nop
   1a620: 00 00 00 00  	v.nop
   1a624: 00 00 00 00 00 00    	m.nop
   1a62a: 69 23 00 12  	l.bc.lz.u32 r13, r4
   1a62e: 01 00        	c.nop
   1a630: 00 00 00 00  	v.nop
   1a634: 00 00 00 00 00 00    	m.nop
   1a63a: 2f 20 01 21  	l.ldi16.s r4, 0x20
   1a63e: 01 00        	c.nop
   1a640: 00 00 00 00  	v.nop
   1a644: 00 00 00 00 00 00    	m.nop
   1a64a: bb 68 84 c0  	l.blt.s r13, r4, 0x2 <0x1a660>
   1a64e: 01 00        	c.nop
   1a650: 00 00 00 00  	v.nop
   1a654: 00 00 00 00 00 00    	m.nop
   1a65a: 2b 48 e1 03  	l.jal r0, 0xe19 <0x287e0>
   1a65e: 01 00        	c.nop
   1a660: 00 00 00 00  	v.nop
   1a664: 00 00 00 00 00 00    	m.nop
   1a66a: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1a66e: 7d 48        	<unknown>
   1a670: 00 00 00 00  	v.nop
   1a674: 00 00 00 00 00 00    	m.nop
   1a67a: 23 21 8d 02  	l.srl     r4, r4, r13
   1a67e: 09 4b        	<unknown>
   1a680: 00 00 00 00  	v.nop
   1a684: 00 00 00 00 00 00    	m.nop
   1a68a: 31 13 c4 16  	l.st.w r4, [sp, 0x16c]
   1a68e: 01 00        	c.nop
   1a690: 00 00 00 00  	v.nop
   1a694: 00 00 00 00 00 00    	m.nop
   1a69a: 25 21 0e 02  	l.orix r4, r4, 0xe
   1a69e: 01 00        	c.nop
   1a6a0: 00 00 00 00  	v.nop
   1a6a4: 00 00 00 00 00 00    	m.nop
   1a6aa: 31 40 c4 00  	l.st.w r4, [r8]
   1a6ae: 01 00        	c.nop
   1a6b0: 00 00 00 00  	v.nop
   1a6b4: 00 00 00 00 00 00    	m.nop
   1a6ba: 31 12 4c 04  	l.ld.w r8, [sp, 0x4c]
   1a6be: 01 00        	c.nop
   1a6c0: 00 00 00 00  	v.nop
   1a6c4: 00 00 00 00 00 00    	m.nop
   1a6ca: 71 15 50 08  	l.ld.w r21, [sp, 0x90]
   1a6ce: 01 00        	c.nop
   1a6d0: 00 00 00 00  	v.nop
   1a6d4: 00 00 00 00 00 00    	m.nop
   1a6da: 31 16 44 12  	l.ld.w r24, [sp, 0x124]
   1a6de: 01 00        	c.nop
   1a6e0: 00 00 00 00  	v.nop
   1a6e4: 00 00 00 00 00 00    	m.nop
   1a6ea: f1 16 40 12  	l.ld.w r27, [sp, 0x120]
   1a6ee: 01 00        	c.nop
   1a6f0: 00 00 00 00  	v.nop
   1a6f4: 00 00 00 00 00 00    	m.nop
   1a6fa: b1 43 40 00  	l.ld.w r14, [r8]
   1a6fe: 01 00        	c.nop
   1a700: 00 00 00 00  	v.nop
   1a704: 00 00 00 00 00 00    	m.nop
   1a70a: 25 71 92 02  	l.xorix r4, r14, 0x12
   1a70e: 01 00        	c.nop
   1a710: 00 00 00 00  	v.nop
   1a714: 00 00 00 00 00 00    	m.nop
   1a71a: 69 21 00 12  	l.bc.lz.u32 r5, r4
   1a71e: 01 00        	c.nop
   1a720: 00 00 00 00  	v.nop
   1a724: 00 00 00 00 00 00    	m.nop
   1a72a: fb 81 85 c0  	l.blt.s r16, r5, 0x7 <0x1a790>
   1a72e: 01 00        	c.nop
   1a730: 00 00 00 00  	v.nop
   1a734: 00 00 00 00 00 00    	m.nop
   1a73a: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1a73e: 01 00        	c.nop
   1a740: 00 00 00 00  	v.nop
   1a744: 00 00 00 00 00 00    	m.nop
   1a74a: 23 21 85 02  	l.srl     r4, r4, r5
   1a74e: 01 00        	c.nop
   1a750: 00 00 00 00  	v.nop
   1a754: 00 00 00 00 00 00    	m.nop
   1a75a: 25 21 0e 02  	l.orix r4, r4, 0xe
   1a75e: 01 00        	c.nop
   1a760: 00 00 00 00  	v.nop
   1a764: 00 00 00 00 00 00    	m.nop
   1a76a: 31 40 c4 00  	l.st.w r4, [r8]
   1a76e: 01 00        	c.nop
   1a770: 00 00 00 00  	v.nop
   1a774: 00 00 00 00 00 00    	m.nop
   1a77a: f1 14 48 06  	l.ld.w r19, [sp, 0x68]
   1a77e: 01 00        	c.nop
   1a780: 00 00 00 00  	v.nop
   1a784: 00 00 00 00 00 00    	m.nop
   1a78a: 2b 68 80 01  	l.jal r0, 0xd <0x1a850>
   1a78e: 01 00        	c.nop
   1a790: 00 00 00 00  	v.nop
   1a794: 00 00 00 00 00 00    	m.nop
   1a79a: f1 43 44 00  	l.ld.w r15, [r8, 0x4]
   1a79e: 01 00        	c.nop
   1a7a0: 00 00 00 00  	v.nop
   1a7a4: 00 00 00 00 00 00    	m.nop
   1a7aa: 25 79 92 02  	l.xorix r4, r15, 0x12
   1a7ae: 01 00        	c.nop
   1a7b0: 00 00 00 00  	v.nop
   1a7b4: 00 00 00 00 00 00    	m.nop
   1a7ba: a9 23 00 12  	l.bc.lz.u32 r14, r4
   1a7be: 01 00        	c.nop
   1a7c0: 00 00 00 00  	v.nop
   1a7c4: 00 00 00 00 00 00    	m.nop
   1a7ca: bb 80 8e 80  	l.bge.s r16, r14, 0x2 <0x1a7e0>
   1a7ce: 01 00        	c.nop
   1a7d0: 00 00 00 00  	v.nop
   1a7d4: 00 00 00 00 00 00    	m.nop
   1a7da: 2b 08 e0 03  	l.jal r0, 0xe01 <0x287e0>
   1a7de: 01 00        	c.nop
   1a7e0: 00 00 00 00  	v.nop
   1a7e4: 00 00 00 00 00 00    	m.nop
   1a7ea: 71 11 44 02  	l.ld.w r5, [sp, 0x24]
   1a7ee: 01 00        	c.nop
   1a7f0: 00 00 00 00  	v.nop
   1a7f4: 00 00 00 00 00 00    	m.nop
   1a7fa: f1 14 48 06  	l.ld.w r19, [sp, 0x68]
   1a7fe: 01 00        	c.nop
   1a800: 00 00 00 00  	v.nop
   1a804: 00 00 00 00 00 00    	m.nop
   1a80a: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1a80e: 01 00        	c.nop
   1a810: 00 00 00 00  	v.nop
   1a814: 00 00 00 00 00 00    	m.nop
   1a81a: 23 21 8e 02  	l.srl     r4, r4, r14
   1a81e: 01 00        	c.nop
   1a820: 00 00 00 00  	v.nop
   1a824: 00 00 00 00 00 00    	m.nop
   1a82a: 25 21 0f 02  	l.orix r4, r4, 0xf
   1a82e: 01 00        	c.nop
   1a830: 00 00 00 00  	v.nop
   1a834: 00 00 00 00 00 00    	m.nop
   1a83a: 31 28 c4 00  	l.st.w r4, [r5]
   1a83e: 01 00        	c.nop
   1a840: 00 00 00 00  	v.nop
   1a844: 00 00 00 00 00 00    	m.nop
   1a84a: 61 71 20 00  	l.addi.u r5, r14, 0x20
   1a84e: 01 00        	c.nop
   1a850: 00 00 00 00  	v.nop
   1a854: 00 00 00 00 00 00    	m.nop
   1a85a: 23 69 09 03  	l.slli r4, r13, 0x9
   1a85e: 7d 59        	<unknown>
   1a860: 00 00 00 00  	v.nop
   1a864: 00 00 00 00 00 00    	m.nop
   1a86a: e3 23 17 18  	<unknown>
   1a86e: 01 00        	c.nop
   1a870: 00 00 00 00  	v.nop
   1a874: 00 00 00 00 00 00    	m.nop
   1a87a: 23 29 09 18  	<unknown>
   1a87e: 01 00        	c.nop
   1a880: 00 00 00 00  	v.nop
   1a884: 00 00 00 00 00 00    	m.nop
   1a88a: 33 02 80 00  	l.movs r8, sip_id
   1a88e: 01 00        	c.nop
   1a890: 00 00 00 00  	v.nop
   1a894: 00 00 00 00 00 00    	m.nop
   1a89a: 31 50 c4 00  	l.st.w r4, [r10]
   1a89e: 01 00        	c.nop
   1a8a0: 00 00 00 00  	v.nop
   1a8a4: 00 00 00 00 00 00    	m.nop
   1a8aa: 33 fa 80 08  	<unknown>
   1a8ae: 01 00        	c.nop
   1a8b0: 00 00 00 00  	v.nop
   1a8b4: 00 00 00 00 00 00    	m.nop
   1a8ba: 33 fa 80 08  	<unknown>
   1a8be: 01 00        	c.nop
   1a8c0: 00 00 00 00  	v.nop
   1a8c4: 00 00 00 00 00 00    	m.nop
   1a8ca: 2f 45 2e 1b  	l.ldi16.u r8, 0xd5d4
   1a8ce: 01 00        	c.nop
   1a8d0: 00 00 00 00  	v.nop
   1a8d4: 00 00 00 00 00 00    	m.nop
   1a8da: 23 79 0b 10  	<unknown>
   1a8de: 01 00        	c.nop
   1a8e0: 00 00 00 00  	v.nop
   1a8e4: 00 00 00 00 00 00    	m.nop
   1a8ea: 31 78 cc 00  	l.st.w r12, [r15]
   1a8ee: 01 00        	c.nop
   1a8f0: 00 00 00 00  	v.nop
   1a8f4: 00 00 00 00 00 00    	m.nop
   1a8fa: 31 20 da 00  	l.st.w r26, [r4]
   1a8fe: 01 00        	c.nop
   1a900: 00 00 00 00  	v.nop
   1a904: 00 00 00 00 00 00    	m.nop
   1a90a: 31 26 da fe  	l.st.w r26, [r4, -0x8]
   1a90e: 01 00        	c.nop
   1a910: 00 00 00 00  	v.nop
   1a914: 00 00 00 00 00 00    	m.nop
   1a91a: 31 22 da 00  	l.st.w r26, [r4, 0x8]
   1a91e: 01 00        	c.nop
   1a920: 00 00 00 00  	v.nop
   1a924: 00 00 00 00 00 00    	m.nop
   1a92a: 31 23 c5 00  	l.st.w r5, [r4, 0xc]
   1a92e: 01 00        	c.nop
   1a930: 00 00 00 00  	v.nop
   1a934: 00 00 00 00 00 00    	m.nop
   1a93a: 31 22 c7 00  	l.st.w r7, [r4, 0x8]
   1a93e: 01 00        	c.nop
   1a940: 00 00 00 00  	v.nop
   1a944: 00 00 00 00 00 00    	m.nop
   1a94a: 6f 40 c0 00  	l.ldi_hi r8, 0x801
   1a94e: 01 00        	c.nop
   1a950: 00 00 00 00  	v.nop
   1a954: 00 00 00 00 00 00    	m.nop
   1a95a: b1 43 40 00  	l.ld.w r14, [r8]
   1a95e: 01 00        	c.nop
   1a960: 00 00 00 00  	v.nop
   1a964: 00 00 00 00 00 00    	m.nop
   1a96a: 25 71 92 02  	l.xorix r4, r14, 0x12
   1a96e: 01 00        	c.nop
   1a970: 00 00 00 00  	v.nop
   1a974: 00 00 00 00 00 00    	m.nop
   1a97a: 29 24 00 12  	l.bc.lz.u32 r16, r4
   1a97e: 01 00        	c.nop
   1a980: 00 00 00 00  	v.nop
   1a984: 00 00 00 00 00 00    	m.nop
   1a98a: 2f 20 01 21  	l.ldi16.s r4, 0x20
   1a98e: 01 00        	c.nop
   1a990: 00 00 00 00  	v.nop
   1a994: 00 00 00 00 00 00    	m.nop
   1a99a: bb 80 84 c0  	l.blt.s r16, r4, 0x2 <0x1a9b0>
   1a99e: 01 00        	c.nop
   1a9a0: 00 00 00 00  	v.nop
   1a9a4: 00 00 00 00 00 00    	m.nop
   1a9aa: 2b 20 de 03  	l.jal r0, 0xde4 <0x287e0>
   1a9ae: 01 00        	c.nop
   1a9b0: 00 00 00 00  	v.nop
   1a9b4: 00 00 00 00 00 00    	m.nop
   1a9ba: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1a9be: 01 00        	c.nop
   1a9c0: 00 00 00 00  	v.nop
   1a9c4: 00 00 00 00 00 00    	m.nop
   1a9ca: 23 21 90 02  	l.srl     r4, r4, r16
   1a9ce: 01 00        	c.nop
   1a9d0: 00 00 00 00  	v.nop
   1a9d4: 00 00 00 00 00 00    	m.nop
   1a9da: 31 12 c4 16  	l.st.w r4, [sp, 0x168]
   1a9de: 01 00        	c.nop
   1a9e0: 00 00 00 00  	v.nop
   1a9e4: 00 00 00 00 00 00    	m.nop
   1a9ea: 25 21 0e 02  	l.orix r4, r4, 0xe
   1a9ee: 01 00        	c.nop
   1a9f0: 00 00 00 00  	v.nop
   1a9f4: 00 00 00 00 00 00    	m.nop
   1a9fa: 31 40 c4 00  	l.st.w r4, [r8]
   1a9fe: 01 00        	c.nop
   1aa00: 00 00 00 00  	v.nop
   1aa04: 00 00 00 00 00 00    	m.nop
   1aa0a: 31 12 4c 04  	l.ld.w r8, [sp, 0x4c]
   1aa0e: 01 00        	c.nop
   1aa10: 00 00 00 00  	v.nop
   1aa14: 00 00 00 00 00 00    	m.nop
   1aa1a: 31 11 d0 0a  	l.st.w r16, [sp, 0xa4]
   1aa1e: 7d 48        	<unknown>
   1aa20: 00 00 00 00  	v.nop
   1aa24: 00 00 00 00 00 00    	m.nop
   1aa2a: 31 12 cf 0e  	l.st.w r15, [sp, 0xe8]
   1aa2e: 01 00        	c.nop
   1aa30: 00 00 00 00  	v.nop
   1aa34: 00 00 00 00 00 00    	m.nop
   1aa3a: 31 16 cd 16  	l.st.w r13, [sp, 0x178]
   1aa3e: 01 00        	c.nop
   1aa40: 00 00 00 00  	v.nop
   1aa44: 00 00 00 00 00 00    	m.nop
   1aa4a: 31 10 c5 0e  	l.st.w r5, [sp, 0xe0]
   1aa4e: 01 00        	c.nop
   1aa50: 00 00 00 00  	v.nop
   1aa54: 00 00 00 00 00 00    	m.nop
   1aa5a: b1 43 40 00  	l.ld.w r14, [r8]
   1aa5e: 01 00        	c.nop
   1aa60: 00 00 00 00  	v.nop
   1aa64: 00 00 00 00 00 00    	m.nop
   1aa6a: 25 71 92 02  	l.xorix r4, r14, 0x12
   1aa6e: 01 00        	c.nop
   1aa70: 00 00 00 00  	v.nop
   1aa74: 00 00 00 00 00 00    	m.nop
   1aa7a: 69 26 00 12  	l.bc.lz.u32 r25, r4
   1aa7e: 01 00        	c.nop
   1aa80: 00 00 00 00  	v.nop
   1aa84: 00 00 00 00 00 00    	m.nop
   1aa8a: 3b 82 99 c0  	l.blt.s r16, r25, 0x8 <0x1ab00>
   1aa8e: 01 00        	c.nop
   1aa90: 00 00 00 00  	v.nop
   1aa94: 00 00 00 00 00 00    	m.nop
   1aa9a: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1aa9e: 81 47        	<unknown>
   1aaa0: 00 00 00 00  	v.nop
   1aaa4: 00 00 00 00 00 00    	m.nop
   1aaaa: 23 21 99 02  	l.srl     r4, r4, r25
   1aaae: e6 82        	<unknown>
   1aab0: 00 00 00 00  	v.nop
   1aab4: 00 00 00 00 00 00    	m.nop
   1aaba: 25 21 0e 02  	l.orix r4, r4, 0xe
   1aabe: 01 00        	c.nop
   1aac0: 00 00 00 00  	v.nop
   1aac4: 00 00 00 00 00 00    	m.nop
   1aaca: 31 40 c4 00  	l.st.w r4, [r8]
   1aace: 01 00        	c.nop
   1aad0: 00 00 00 00  	v.nop
   1aad4: 00 00 00 00 00 00    	m.nop
   1aada: 71 13 44 06  	l.ld.w r13, [sp, 0x64]
   1aade: 01 00        	c.nop
   1aae0: 00 00 00 00  	v.nop
   1aae4: 00 00 00 00 00 00    	m.nop
   1aaea: 31 11 44 0a  	l.ld.w r4, [sp, 0xa4]
   1aaee: 01 00        	c.nop
   1aaf0: 00 00 00 00  	v.nop
   1aaf4: 00 00 00 00 00 00    	m.nop
   1aafa: 2b 70 80 01  	l.jal r0, 0xe <0x1abd0>
   1aafe: 01 00        	c.nop
   1ab00: 00 00 00 00  	v.nop
   1ab04: 00 00 00 00 00 00    	m.nop
   1ab0a: f1 43 44 00  	l.ld.w r15, [r8, 0x4]
   1ab0e: 01 00        	c.nop
   1ab10: 00 00 00 00  	v.nop
   1ab14: 00 00 00 00 00 00    	m.nop
   1ab1a: 25 79 92 02  	l.xorix r4, r15, 0x12
   1ab1e: 01 00        	c.nop
   1ab20: 00 00 00 00  	v.nop
   1ab24: 00 00 00 00 00 00    	m.nop
   1ab2a: a9 23 00 12  	l.bc.lz.u32 r14, r4
   1ab2e: 01 00        	c.nop
   1ab30: 00 00 00 00  	v.nop
   1ab34: 00 00 00 00 00 00    	m.nop
   1ab3a: bb 80 8e 80  	l.bge.s r16, r14, 0x2 <0x1ab50>
   1ab3e: 01 00        	c.nop
   1ab40: 00 00 00 00  	v.nop
   1ab44: 00 00 00 00 00 00    	m.nop
   1ab4a: 2b 50 dc 03  	l.jal r0, 0xdca <0x287e0>
   1ab4e: 01 00        	c.nop
   1ab50: 00 00 00 00  	v.nop
   1ab54: 00 00 00 00 00 00    	m.nop
   1ab5a: 71 11 44 02  	l.ld.w r5, [sp, 0x24]
   1ab5e: 01 00        	c.nop
   1ab60: 00 00 00 00  	v.nop
   1ab64: 00 00 00 00 00 00    	m.nop
   1ab6a: 2f 20 80 30  	l.ldi_hi.m1 r4, 0x8000
   1ab6e: 01 00        	c.nop
   1ab70: 00 00 00 00  	v.nop
   1ab74: 00 00 00 00 00 00    	m.nop
   1ab7a: 23 21 8e 02  	l.srl     r4, r4, r14
   1ab7e: 01 00        	c.nop
   1ab80: 00 00 00 00  	v.nop
   1ab84: 00 00 00 00 00 00    	m.nop
   1ab8a: 25 21 0f 02  	l.orix r4, r4, 0xf
   1ab8e: 81 47        	<unknown>
   1ab90: 00 00 00 00  	v.nop
   1ab94: 00 00 00 00 00 00    	m.nop
   1ab9a: 71 13 44 06  	l.ld.w r13, [sp, 0x64]
   1ab9e: 01 00        	c.nop
   1aba0: 00 00 00 00  	v.nop
   1aba4: 00 00 00 00 00 00    	m.nop
   1abaa: 31 28 c4 00  	l.st.w r4, [r5]
   1abae: 01 00        	c.nop
   1abb0: 00 00 00 00  	v.nop
   1abb4: 00 00 00 00 00 00    	m.nop
   1abba: 31 11 44 0a  	l.ld.w r4, [sp, 0xa4]
   1abbe: 01 00        	c.nop
   1abc0: 00 00 00 00  	v.nop
   1abc4: 00 00 00 00 00 00    	m.nop
   1abca: 61 71 20 00  	l.addi.u r5, r14, 0x20
   1abce: 01 00        	c.nop
   1abd0: 00 00 00 00  	v.nop
   1abd4: 00 00 00 00 00 00    	m.nop
   1abda: 00 00        	s.nop16
   1abdc: 00 00        	s.nop16
   1abde: 26 02        	<unknown>
   1abe0: 00 00 00 00  	v.nop
   1abe4: 00 00 00 00 00 00    	m.nop
   1abea: a3 23 17 18  	<unknown>
   1abee: 01 00        	c.nop
   1abf0: 00 00 00 00  	v.nop
   1abf4: 00 00 00 00 00 00    	m.nop
   1abfa: 31 11 5c 02  	l.ld.w r4, [sp, 0x3c]
   1abfe: 01 00        	c.nop
   1ac00: 00 00 00 00  	v.nop
   1ac04: 00 00 00 00 00 00    	m.nop
   1ac0a: 23 72 0b 10  	<unknown>
   1ac0e: 01 00        	c.nop
   1ac10: 00 00 00 00  	v.nop
   1ac14: 00 00 00 00 00 00    	m.nop
   1ac1a: 23 24 06 19  	<unknown>
   1ac1e: 01 00        	c.nop
   1ac20: 00 00 00 00  	v.nop
   1ac24: 00 00 00 00 00 00    	m.nop
   1ac2a: 23 29 09 18  	<unknown>
   1ac2e: 01 00        	c.nop
   1ac30: 00 00 00 00  	v.nop
   1ac34: 00 00 00 00 00 00    	m.nop
   1ac3a: 73 02 80 00  	l.movs r9, sip_id
   1ac3e: 01 00        	c.nop
   1ac40: 00 00 00 00  	v.nop
   1ac44: 00 00 00 00 00 00    	m.nop
   1ac4a: 31 50 c4 00  	l.st.w r4, [r10]
   1ac4e: 01 00        	c.nop
   1ac50: 00 00 00 00  	v.nop
   1ac54: 00 00 00 00 00 00    	m.nop
   1ac5a: 33 f9 80 08  	<unknown>
   1ac5e: 01 00        	c.nop
   1ac60: 00 00 00 00  	v.nop
   1ac64: 00 00 00 00 00 00    	m.nop
   1ac6a: 33 f9 80 08  	<unknown>
   1ac6e: 01 00        	c.nop
   1ac70: 00 00 00 00  	v.nop
   1ac74: 00 00 00 00 00 00    	m.nop
   1ac7a: 31 70 cc 00  	l.st.w r12, [r14]
   1ac7e: 01 00        	c.nop
   1ac80: 00 00 00 00  	v.nop
   1ac84: 00 00 00 00 00 00    	m.nop
   1ac8a: 31 40 da 00  	l.st.w r26, [r8]
   1ac8e: 01 00        	c.nop
   1ac90: 00 00 00 00  	v.nop
   1ac94: 00 00 00 00 00 00    	m.nop
   1ac9a: 31 46 da fe  	l.st.w r26, [r8, -0x8]
   1ac9e: 01 00        	c.nop
   1aca0: 00 00 00 00  	v.nop
   1aca4: 00 00 00 00 00 00    	m.nop
   1acaa: 31 42 da 00  	l.st.w r26, [r8, 0x8]
   1acae: 01 00        	c.nop
   1acb0: 00 00 00 00  	v.nop
   1acb4: 00 00 00 00 00 00    	m.nop
   1acba: 31 10 c5 0a  	l.st.w r5, [sp, 0xa0]
   1acbe: 01 00        	c.nop
   1acc0: 00 00 00 00  	v.nop
   1acc4: 00 00 00 00 00 00    	m.nop
   1acca: 31 43 c5 00  	l.st.w r5, [r8, 0xc]
   1acce: 01 00        	c.nop
   1acd0: 00 00 00 00  	v.nop
   1acd4: 00 00 00 00 00 00    	m.nop
   1acda: 31 42 c7 00  	l.st.w r7, [r8, 0x8]
   1acde: 01 00        	c.nop
   1ace0: 00 00 00 00  	v.nop
   1ace4: 00 00 00 00 00 00    	m.nop
   1acea: 31 11 d0 1a  	l.st.w r16, [sp, 0x1a4]
   1acee: 01 00        	c.nop
   1acf0: 00 00 00 00  	v.nop
   1acf4: 00 00 00 00 00 00    	m.nop
   1acfa: bb 68 90 c0  	l.blt.s r13, r16, 0x2 <0x1ad10>
   1acfe: 01 00        	c.nop
   1ad00: 00 00 00 00  	v.nop
   1ad04: 00 00 00 00 00 00    	m.nop
   1ad0a: 2b 60 c8 03  	l.jal r0, 0xc8c <0x275c0>
   1ad0e: 01 00        	c.nop
   1ad10: 00 00 00 00  	v.nop
   1ad14: 00 00 00 00 00 00    	m.nop
   1ad1a: 2f 28 10 01  	l.ldi16.u r5, 0x200
   1ad1e: 71 42        	<unknown>
   1ad20: 00 00 00 00  	v.nop
   1ad24: 00 00 00 00 00 00    	m.nop
   1ad2a: 2f 3a 10 21  	l.ldi16.s r7, 0x208
   1ad2e: 01 00        	c.nop
   1ad30: 00 00 00 00  	v.nop
   1ad34: 00 00 00 00 00 00    	m.nop
   1ad3a: 23 71 04 10  	<unknown>
   1ad3e: 01 00        	c.nop
   1ad40: 00 00 00 00  	v.nop
   1ad44: 00 00 00 00 00 00    	m.nop
   1ad4a: 2f 28 80 00  	l.ldi_hi r5, 0x0
   1ad4e: 01 00        	c.nop
   1ad50: 00 00 00 00  	v.nop
   1ad54: 00 00 00 00 00 00    	m.nop
   1ad5a: 31 20 cf 00  	l.st.w r15, [r4]
   1ad5e: 01 00        	c.nop
   1ad60: 00 00 00 00  	v.nop
   1ad64: 00 00 00 00 00 00    	m.nop
   1ad6a: 31 27 cf fe  	l.st.w r15, [r4, -0x4]
   1ad6e: 01 00        	c.nop
   1ad70: 00 00 00 00  	v.nop
   1ad74: 00 00 00 00 00 00    	m.nop
   1ad7a: 31 15 c5 22  	l.st.w r5, [sp, 0x234]
   1ad7e: 01 00        	c.nop
   1ad80: 00 00 00 00  	v.nop
   1ad84: 00 00 00 00 00 00    	m.nop
   1ad8a: 31 22 c5 00  	l.st.w r5, [r4, 0x8]
   1ad8e: 01 00        	c.nop
   1ad90: 00 00 00 00  	v.nop
   1ad94: 00 00 00 00 00 00    	m.nop
   1ad9a: 2f 38 81 00  	l.ldi_hi r7, 0x20
   1ad9e: 01 00        	c.nop
   1ada0: 00 00 00 00  	v.nop
   1ada4: 00 00 00 00 00 00    	m.nop
   1adaa: 2f 28 08 21  	l.ldi16.s r5, 0x100
   1adae: 01 00        	c.nop
   1adb0: 00 00 00 00  	v.nop
   1adb4: 00 00 00 00 00 00    	m.nop
   1adba: 2f 40 fc 3e  	l.ldi_hi.m1 r8, 0xff80
   1adbe: 01 00        	c.nop
   1adc0: 00 00 00 00  	v.nop
   1adc4: 00 00 00 00 00 00    	m.nop
   1adca: 31 21 cf 00  	l.st.w r15, [r4, 0x4]
   1adce: 01 00        	c.nop
   1add0: 00 00 00 00  	v.nop
   1add4: 00 00 00 00 00 00    	m.nop
   1adda: 31 25 c5 00  	l.st.w r5, [r4, 0x14]
   1adde: 91 42        	<unknown>
   1ade0: 00 00 00 00  	v.nop
   1ade4: 00 00 00 00 00 00    	m.nop
   1adea: 31 27 c8 0a  	l.st.w r8, [r4, 0xbc]
   1adee: 01 00        	c.nop
   1adf0: 00 00 00 00  	v.nop
   1adf4: 00 00 00 00 00 00    	m.nop
   1adfa: 31 26 c7 fe  	l.st.w r7, [r4, -0x8]
   1adfe: 01 00        	c.nop
   1ae00: 00 00 00 00  	v.nop
   1ae04: 00 00 00 00 00 00    	m.nop
   1ae0a: 33 b9 83 00  	l.movs r4, md23_wrk
   1ae0e: 01 00        	c.nop
   1ae10: 00 00 00 00  	v.nop
   1ae14: 00 00 00 00 00 00    	m.nop
   1ae1a: ef bf 7f 1f  	l.ldi16.u r23, 0xffff
   1ae1e: 01 00        	c.nop
   1ae20: 00 00 00 00  	v.nop
   1ae24: 00 00 00 00 00 00    	m.nop
   1ae2a: 25 21 85 01  	l.andix r4, r4, 0x5
   1ae2e: 01 00        	c.nop
   1ae30: 00 00 00 00  	v.nop
   1ae34: 00 00 00 00 00 00    	m.nop
   1ae3a: ef bf 87 00  	l.ldi_hi r23, 0xff
   1ae3e: 01 00        	c.nop
   1ae40: 00 00 00 00  	v.nop
   1ae44: 00 00 00 00 00 00    	m.nop
   1ae4a: af ef 80 20  	l.ldi_hi.m1 r29, 0x1e
   1ae4e: 01 00        	c.nop
   1ae50: 00 00 00 00  	v.nop
   1ae54: 00 00 00 00 00 00    	m.nop
   1ae5a: 3b 24 00 00  	<unknown>
   1ae5e: 01 00        	c.nop
   1ae60: 00 00 00 00  	v.nop
   1ae64: 00 00 00 00 00 00    	m.nop
   1ae6a: 2f 22 04 21  	l.ldi16.s r4, 0x88
   1ae6e: 01 00        	c.nop
   1ae70: 00 00 00 00  	v.nop
   1ae74: 00 00 00 00 00 00    	m.nop
   1ae7a: 23 71 04 10  	<unknown>
   1ae7e: 01 00        	c.nop
   1ae80: 00 00 00 00  	v.nop
   1ae84: 00 00 00 00 00 00    	m.nop
   1ae8a: 31 22 40 00  	l.ld.w r8, [r4]
   1ae8e: 01 00        	c.nop
   1ae90: 00 00 00 00  	v.nop
   1ae94: 00 00 00 00 00 00    	m.nop
   1ae9a: 73 a2 83 00  	l.movs r9, md20_wrk
   1ae9e: 01 00        	c.nop
   1aea0: 00 00 00 00  	v.nop
   1aea4: 00 00 00 00 00 00    	m.nop
   1aeaa: 2f 28 f0 3e  	l.ldi_hi.m1 r5, 0xfe00
   1aeae: 86 04        	<unknown>
   1aeb0: 00 00 00 00  	v.nop
   1aeb4: 00 00 00 00 00 00    	m.nop
   1aeba: 65 4a 85 01  	l.andix r9, r9, 0x5
   1aebe: 01 00        	c.nop
   1aec0: 00 00 00 00  	v.nop
   1aec4: 00 00 00 00 00 00    	m.nop
   1aeca: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   1aece: 01 00        	c.nop
   1aed0: 00 00 00 00  	v.nop
   1aed4: 00 00 00 00 00 00    	m.nop
   1aeda: b3 02 80 00  	l.movs r10, sip_id
   1aede: 01 00        	c.nop
   1aee0: 00 00 00 00  	v.nop
   1aee4: 00 00 00 00 00 00    	m.nop
   1aeea: 6f 28 80 00  	l.ldi_hi r5, 0x1
   1aeee: 3e 05        	<unknown>
   1aef0: 00 00 00 00  	v.nop
   1aef4: 00 00 00 00 00 00    	m.nop
   1aefa: 25 42 85 01  	l.andix r8, r8, 0x5
   1aefe: 01 00        	c.nop
   1af00: 00 00 00 00  	v.nop
   1af04: 00 00 00 00 00 00    	m.nop
   1af0a: e1 ba 01 00  	l.addi.u r11, r23, 0x1
   1af0e: 45 8c        	<unknown>
   1af10: 00 00 00 00  	v.nop
   1af14: 00 00 00 00 00 00    	m.nop
   1af1a: 00 00        	s.nop16
   1af1c: 00 00        	s.nop16
   1af1e: 6d 8d        	<unknown>
   1af20: 00 00 00 00  	v.nop
   1af24: 00 00 00 00 00 00    	m.nop
   1af2a: 00 00        	s.nop16
   1af2c: 00 00        	s.nop16
   1af2e: 49 8c        	<unknown>
   1af30: 00 00 00 00  	v.nop
   1af34: 00 00 00 00 00 00    	m.nop
   1af3a: 25 42 1d 02  	l.orix r8, r8, 0x1d
   1af3e: 01 00        	c.nop
   1af40: 00 00 00 00  	v.nop
   1af44: 00 00 00 00 00 00    	m.nop
   1af4a: 31 20 c8 00  	l.st.w r8, [r4]
   1af4e: 01 00        	c.nop
   1af50: 00 00 00 00  	v.nop
   1af54: 00 00 00 00 00 00    	m.nop
   1af5a: 2f 2d 01 21  	l.ldi16.s r5, 0x34
   1af5e: 01 00        	c.nop
   1af60: 00 00 00 00  	v.nop
   1af64: 00 00 00 00 00 00    	m.nop
   1af6a: 63 72 05 10  	<unknown>
   1af6e: 01 00        	c.nop
   1af70: 00 00 00 00  	v.nop
   1af74: 00 00 00 00 00 00    	m.nop
   1af7a: 33 02 80 00  	l.movs r8, sip_id
   1af7e: 01 00        	c.nop
   1af80: 00 00 00 00  	v.nop
   1af84: 00 00 00 00 00 00    	m.nop
   1af8a: 31 48 da 00  	l.st.w r26, [r9]
   1af8e: 01 00        	c.nop
   1af90: 00 00 00 00  	v.nop
   1af94: 00 00 00 00 00 00    	m.nop
   1af9a: 71 11 40 0a  	l.ld.w r5, [sp, 0xa0]
   1af9e: 01 00        	c.nop
   1afa0: 00 00 00 00  	v.nop
   1afa4: 00 00 00 00 00 00    	m.nop
   1afaa: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1afae: 01 00        	c.nop
   1afb0: 00 00 00 00  	v.nop
   1afb4: 00 00 00 00 00 00    	m.nop
   1afba: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1afbe: 01 00        	c.nop
   1afc0: 00 00 00 00  	v.nop
   1afc4: 00 00 00 00 00 00    	m.nop
   1afca: 21 21 00 01  	l.addi.u r4, r4, 0x100
   1afce: 01 00        	c.nop
   1afd0: 00 00 00 00  	v.nop
   1afd4: 00 00 00 00 00 00    	m.nop
   1afda: 2d 28 20 00  	l.wait.dec r5
   1afde: 01 00        	c.nop
   1afe0: 00 00 00 00  	v.nop
   1afe4: 00 00 00 00 00 00    	m.nop
   1afea: 31 4a cf fe  	l.st.w r15, [r9, -0x18]
   1afee: 91 42        	<unknown>
   1aff0: 00 00 00 00  	v.nop
   1aff4: 00 00 00 00 00 00    	m.nop
   1affa: 31 49 cf fe  	l.st.w r15, [r9, -0x1c]
   1affe: 01 00        	c.nop
   1b000: 00 00 00 00  	v.nop
   1b004: 00 00 00 00 00 00    	m.nop
   1b00a: 31 14 c4 22  	l.st.w r4, [sp, 0x230]
   1b00e: 01 00        	c.nop
   1b010: 00 00 00 00  	v.nop
   1b014: 00 00 00 00 00 00    	m.nop
   1b01a: 31 4c c4 fe  	l.st.w r4, [r9, -0x10]
   1b01e: 01 00        	c.nop
   1b020: 00 00 00 00  	v.nop
   1b024: 00 00 00 00 00 00    	m.nop
   1b02a: 2f 20 08 21  	l.ldi16.s r4, 0x100
   1b02e: 01 00        	c.nop
   1b030: 00 00 00 00  	v.nop
   1b034: 00 00 00 00 00 00    	m.nop
   1b03a: 31 4b cf fe  	l.st.w r15, [r9, -0x14]
   1b03e: 01 00        	c.nop
   1b040: 00 00 00 00  	v.nop
   1b044: 00 00 00 00 00 00    	m.nop
   1b04a: 31 4f c4 fe  	l.st.w r4, [r9, -0x4]
   1b04e: 01 00        	c.nop
   1b050: 00 00 00 00  	v.nop
   1b054: 00 00 00 00 00 00    	m.nop
   1b05a: 2f 22 10 21  	l.ldi16.s r4, 0x208
   1b05e: 01 00        	c.nop
   1b060: 00 00 00 00  	v.nop
   1b064: 00 00 00 00 00 00    	m.nop
   1b06a: 2f 20 81 00  	l.ldi_hi r4, 0x20
   1b06e: 01 00        	c.nop
   1b070: 00 00 00 00  	v.nop
   1b074: 00 00 00 00 00 00    	m.nop
   1b07a: 31 49 cf 0a  	l.st.w r15, [r9, 0xa4]
   1b07e: 01 00        	c.nop
   1b080: 00 00 00 00  	v.nop
   1b084: 00 00 00 00 00 00    	m.nop
   1b08a: 31 48 c4 fe  	l.st.w r4, [r9, -0x20]
   1b08e: 01 00        	c.nop
   1b090: 00 00 00 00  	v.nop
   1b094: 00 00 00 00 00 00    	m.nop
   1b09a: 33 b9 83 00  	l.movs r4, md23_wrk
   1b09e: 01 00        	c.nop
   1b0a0: 00 00 00 00  	v.nop
   1b0a4: 00 00 00 00 00 00    	m.nop
   1b0aa: f1 11 44 04  	l.ld.w r7, [sp, 0x44]
   1b0ae: 01 00        	c.nop
   1b0b0: 00 00 00 00  	v.nop
   1b0b4: 00 00 00 00 00 00    	m.nop
   1b0ba: 25 21 85 01  	l.andix r4, r4, 0x5
   1b0be: 01 00        	c.nop
   1b0c0: 00 00 00 00  	v.nop
   1b0c4: 00 00 00 00 00 00    	m.nop
   1b0ca: 3b 22 00 40  	<unknown>
   1b0ce: 01 00        	c.nop
   1b0d0: 00 00 00 00  	v.nop
   1b0d4: 00 00 00 00 00 00    	m.nop
   1b0da: 31 11 da 0e  	l.st.w r26, [sp, 0xe4]
   1b0de: 01 00        	c.nop
   1b0e0: 00 00 00 00  	v.nop
   1b0e4: 00 00 00 00 00 00    	m.nop
   1b0ea: fb 3d 0f 00  	<unknown>
   1b0ee: 01 00        	c.nop
   1b0f0: 00 00 00 00  	v.nop
   1b0f4: 00 00 00 00 00 00    	m.nop
   1b0fa: 31 11 cf 0e  	l.st.w r15, [sp, 0xe4]
   1b0fe: 01 00        	c.nop
   1b100: 00 00 00 00  	v.nop
   1b104: 00 00 00 00 00 00    	m.nop
   1b10a: 31 12 58 06  	l.ld.w r8, [sp, 0x78]
   1b10e: 7d 52        	<unknown>
   1b110: 00 00 00 00  	v.nop
   1b114: 00 00 00 00 00 00    	m.nop
   1b11a: 31 10 da 14  	l.st.w r26, [sp, 0x140]
   1b11e: 01 00        	c.nop
   1b120: 00 00 00 00  	v.nop
   1b124: 00 00 00 00 00 00    	m.nop
   1b12a: bb dd 0f 40  	<unknown>
   1b12e: 01 00        	c.nop
   1b130: 00 00 00 00  	v.nop
   1b134: 00 00 00 00 00 00    	m.nop
   1b13a: 2b 30 81 01  	l.jal r0, 0x16 <0x1b290>
   1b13e: 01 00        	c.nop
   1b140: 00 00 00 00  	v.nop
   1b144: 00 00 00 00 00 00    	m.nop
   1b14a: 2f 22 04 21  	l.ldi16.s r4, 0x88
   1b14e: 01 00        	c.nop
   1b150: 00 00 00 00  	v.nop
   1b154: 00 00 00 00 00 00    	m.nop
   1b15a: 23 71 04 10  	<unknown>
   1b15e: 01 00        	c.nop
   1b160: 00 00 00 00  	v.nop
   1b164: 00 00 00 00 00 00    	m.nop
   1b16a: 31 22 40 00  	l.ld.w r8, [r4]
   1b16e: 01 00        	c.nop
   1b170: 00 00 00 00  	v.nop
   1b174: 00 00 00 00 00 00    	m.nop
   1b17a: b3 a2 83 00  	l.movs r10, md20_wrk
   1b17e: 01 00        	c.nop
   1b180: 00 00 00 00  	v.nop
   1b184: 00 00 00 00 00 00    	m.nop
   1b18a: 2f 28 f0 3e  	l.ldi_hi.m1 r5, 0xfe00
   1b18e: 06 05        	<unknown>
   1b190: 00 00 00 00  	v.nop
   1b194: 00 00 00 00 00 00    	m.nop
   1b19a: a5 52 85 01  	l.andix r10, r10, 0x5
   1b19e: 01 00        	c.nop
   1b1a0: 00 00 00 00  	v.nop
   1b1a4: 00 00 00 00 00 00    	m.nop
   1b1aa: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   1b1ae: 01 00        	c.nop
   1b1b0: 00 00 00 00  	v.nop
   1b1b4: 00 00 00 00 00 00    	m.nop
   1b1ba: f3 02 80 00  	l.movs r11, sip_id
   1b1be: 01 00        	c.nop
   1b1c0: 00 00 00 00  	v.nop
   1b1c4: 00 00 00 00 00 00    	m.nop
   1b1ca: 6f 28 80 00  	l.ldi_hi r5, 0x1
   1b1ce: be 05        	<unknown>
   1b1d0: 00 00 00 00  	v.nop
   1b1d4: 00 00 00 00 00 00    	m.nop
   1b1da: 25 42 85 01  	l.andix r8, r8, 0x5
   1b1de: 01 00        	c.nop
   1b1e0: 00 00 00 00  	v.nop
   1b1e4: 00 00 00 00 00 00    	m.nop
   1b1ea: 21 bb 01 00  	l.addi.u r12, r23, 0x1
   1b1ee: 49 8c        	<unknown>
   1b1f0: 00 00 00 00  	v.nop
   1b1f4: 00 00 00 00 00 00    	m.nop
   1b1fa: 00 00        	s.nop16
   1b1fc: 00 00        	s.nop16
   1b1fe: f1 8d        	<unknown>
   1b200: 00 00 00 00  	v.nop
   1b204: 00 00 00 00 00 00    	m.nop
   1b20a: 00 00        	s.nop16
   1b20c: 00 00        	s.nop16
   1b20e: 4d 8c        	<unknown>
   1b210: 00 00 00 00  	v.nop
   1b214: 00 00 00 00 00 00    	m.nop
   1b21a: 25 42 1d 02  	l.orix r8, r8, 0x1d
   1b21e: 01 00        	c.nop
   1b220: 00 00 00 00  	v.nop
   1b224: 00 00 00 00 00 00    	m.nop
   1b22a: 31 20 c8 00  	l.st.w r8, [r4]
   1b22e: 01 00        	c.nop
   1b230: 00 00 00 00  	v.nop
   1b234: 00 00 00 00 00 00    	m.nop
   1b23a: 31 11 da 0e  	l.st.w r26, [sp, 0xe4]
   1b23e: 01 00        	c.nop
   1b240: 00 00 00 00  	v.nop
   1b244: 00 00 00 00 00 00    	m.nop
   1b24a: fb 3a 6f 7f  	<unknown>
   1b24e: 01 00        	c.nop
   1b250: 00 00 00 00  	v.nop
   1b254: 00 00 00 00 00 00    	m.nop
   1b25a: 31 12 58 06  	l.ld.w r8, [sp, 0x78]
   1b25e: 7d 52        	<unknown>
   1b260: 00 00 00 00  	v.nop
   1b264: 00 00 00 00 00 00    	m.nop
   1b26a: 31 10 da 14  	l.st.w r26, [sp, 0x140]
   1b26e: 01 00        	c.nop
   1b270: 00 00 00 00  	v.nop
   1b274: 00 00 00 00 00 00    	m.nop
   1b27a: bb d8 0f 00  	<unknown>
   1b27e: 01 00        	c.nop
   1b280: 00 00 00 00  	v.nop
   1b284: 00 00 00 00 00 00    	m.nop
   1b28a: 31 10 cf 14  	l.st.w r15, [sp, 0x140]
   1b28e: 01 00        	c.nop
   1b290: 00 00 00 00  	v.nop
   1b294: 00 00 00 00 00 00    	m.nop
   1b29a: 31 14 c4 0e  	l.st.w r4, [sp, 0xf0]
   1b29e: 01 00        	c.nop
   1b2a0: 00 00 00 00  	v.nop
   1b2a4: 00 00 00 00 00 00    	m.nop
   1b2aa: 31 11 54 06  	l.ld.w r4, [sp, 0x74]
   1b2ae: 01 00        	c.nop
   1b2b0: 00 00 00 00  	v.nop
   1b2b4: 00 00 00 00 00 00    	m.nop
   1b2ba: ef 2f 7f 1f  	l.ldi16.u r5, 0xffff
   1b2be: 01 00        	c.nop
   1b2c0: 00 00 00 00  	v.nop
   1b2c4: 00 00 00 00 00 00    	m.nop
   1b2ca: 2f cd 08 21  	l.ldi16.s r25, 0x114
   1b2ce: 96 83        	<unknown>
   1b2d0: 00 00 00 00  	v.nop
   1b2d4: 00 00 00 00 00 00    	m.nop
   1b2da: 65 41 87 01  	l.andix r5, r8, 0x7
   1b2de: 01 00        	c.nop
   1b2e0: 00 00 00 00  	v.nop
   1b2e4: 00 00 00 00 00 00    	m.nop
   1b2ea: 31 12 c5 1c  	l.st.w r5, [sp, 0x1c8]
   1b2ee: 01 00        	c.nop
   1b2f0: 00 00 00 00  	v.nop
   1b2f4: 00 00 00 00 00 00    	m.nop
   1b2fa: 25 21 97 01  	l.andix r4, r4, 0x17
   1b2fe: 01 00        	c.nop
   1b300: 00 00 00 00  	v.nop
   1b304: 00 00 00 00 00 00    	m.nop
   1b30a: 31 11 c4 1c  	l.st.w r4, [sp, 0x1c4]
   1b30e: 01 00        	c.nop
   1b310: 00 00 00 00  	v.nop
   1b314: 00 00 00 00 00 00    	m.nop
   1b31a: 31 11 48 04  	l.ld.w r4, [sp, 0x48]
   1b31e: 01 00        	c.nop
   1b320: 00 00 00 00  	v.nop
   1b324: 00 00 00 00 00 00    	m.nop
   1b32a: 2f 40 10 01  	l.ldi16.u r8, 0x200
   1b32e: 01 00        	c.nop
   1b330: 00 00 00 00  	v.nop
   1b334: 00 00 00 00 00 00    	m.nop
   1b33a: 2f 40 80 00  	l.ldi_hi r8, 0x0
   1b33e: 01 00        	c.nop
   1b340: 00 00 00 00  	v.nop
   1b344: 00 00 00 00 00 00    	m.nop
   1b34a: 2f c8 81 10  	l.ldi_hi r25, 0x8020
   1b34e: 01 00        	c.nop
   1b350: 00 00 00 00  	v.nop
   1b354: 00 00 00 00 00 00    	m.nop
   1b35a: 25 21 87 01  	l.andix r4, r4, 0x7
   1b35e: 01 00        	c.nop
   1b360: 00 00 00 00  	v.nop
   1b364: 00 00 00 00 00 00    	m.nop
   1b36a: 31 10 c4 1c  	l.st.w r4, [sp, 0x1c0]
   1b36e: 01 00        	c.nop
   1b370: 00 00 00 00  	v.nop
   1b374: 00 00 00 00 00 00    	m.nop
   1b37a: 25 31 97 01  	l.andix r4, r6, 0x17
   1b37e: 01 00        	c.nop
   1b380: 00 00 00 00  	v.nop
   1b384: 00 00 00 00 00 00    	m.nop
   1b38a: 31 17 c4 1a  	l.st.w r4, [sp, 0x1bc]
   1b38e: 01 00        	c.nop
   1b390: 00 00 00 00  	v.nop
   1b394: 00 00 00 00 00 00    	m.nop
   1b39a: 33 01 80 00  	l.movs r4, sip_id
   1b39e: 01 00        	c.nop
   1b3a0: 00 00 00 00  	v.nop
   1b3a4: 00 00 00 00 00 00    	m.nop
   1b3aa: 31 48 da 00  	l.st.w r26, [r9]
   1b3ae: 01 00        	c.nop
   1b3b0: 00 00 00 00  	v.nop
   1b3b4: 00 00 00 00 00 00    	m.nop
   1b3ba: 31 11 40 0a  	l.ld.w r4, [sp, 0xa0]
   1b3be: 01 00        	c.nop
   1b3c0: 00 00 00 00  	v.nop
   1b3c4: 00 00 00 00 00 00    	m.nop
   1b3ca: 2f 48 10 01  	l.ldi16.u r9, 0x200
   1b3ce: 01 00        	c.nop
   1b3d0: 00 00 00 00  	v.nop
   1b3d4: 00 00 00 00 00 00    	m.nop
   1b3da: 2f 48 80 00  	l.ldi_hi r9, 0x0
   1b3de: 01 00        	c.nop
   1b3e0: 00 00 00 00  	v.nop
   1b3e4: 00 00 00 00 00 00    	m.nop
   1b3ea: 2d 20 20 00  	l.wait.dec r4
   1b3ee: 01 00        	c.nop
   1b3f0: 00 00 00 00  	v.nop
   1b3f4: 00 00 00 00 00 00    	m.nop
   1b3fa: 31 11 5c 02  	l.ld.w r4, [sp, 0x3c]
   1b3fe: 01 00        	c.nop
   1b400: 00 00 00 00  	v.nop
   1b404: 00 00 00 00 00 00    	m.nop
   1b40a: b1 11 54 02  	l.ld.w r6, [sp, 0x34]
   1b40e: 01 00        	c.nop
   1b410: 00 00 00 00  	v.nop
   1b414: 00 00 00 00 00 00    	m.nop
   1b41a: 31 13 c9 22  	l.st.w r9, [sp, 0x22c]
   1b41e: 01 00        	c.nop
   1b420: 00 00 00 00  	v.nop
   1b424: 00 00 00 00 00 00    	m.nop
   1b42a: 31 17 c8 14  	l.st.w r8, [sp, 0x15c]
   1b42e: 01 00        	c.nop
   1b430: 00 00 00 00  	v.nop
   1b434: 00 00 00 00 00 00    	m.nop
   1b43a: 63 21 02 03  	l.slli r5, r4, 0x2
   1b43e: 01 00        	c.nop
   1b440: 00 00 00 00  	v.nop
   1b444: 00 00 00 00 00 00    	m.nop
   1b44a: 31 10 c5 18  	l.st.w r5, [sp, 0x180]
   1b44e: 01 00        	c.nop
   1b450: 00 00 00 00  	v.nop
   1b454: 00 00 00 00 00 00    	m.nop
   1b45a: 65 21 97 01  	l.andix r5, r4, 0x17
   1b45e: 01 00        	c.nop
   1b460: 00 00 00 00  	v.nop
   1b464: 00 00 00 00 00 00    	m.nop
   1b46a: 31 14 c5 0a  	l.st.w r5, [sp, 0xb0]
   1b46e: 01 00        	c.nop
   1b470: 00 00 00 00  	v.nop
   1b474: 00 00 00 00 00 00    	m.nop
   1b47a: 71 11 54 04  	l.ld.w r5, [sp, 0x54]
   1b47e: 01 00        	c.nop
   1b480: 00 00 00 00  	v.nop
   1b484: 00 00 00 00 00 00    	m.nop
   1b48a: 23 31 04 19  	<unknown>
   1b48e: 01 00        	c.nop
   1b490: 00 00 00 00  	v.nop
   1b494: 00 00 00 00 00 00    	m.nop
   1b49a: a5 37 97 01  	l.andix r30, r6, 0x17
   1b49e: 01 00        	c.nop
   1b4a0: 00 00 00 00  	v.nop
   1b4a4: 00 00 00 00 00 00    	m.nop
   1b4aa: 31 15 de 0e  	l.st.w r30, [sp, 0xf4]
   1b4ae: 01 00        	c.nop
   1b4b0: 00 00 00 00  	v.nop
   1b4b4: 00 00 00 00 00 00    	m.nop
   1b4ba: 31 15 c4 18  	l.st.w r4, [sp, 0x194]
   1b4be: 01 00        	c.nop
   1b4c0: 00 00 00 00  	v.nop
   1b4c4: 00 00 00 00 00 00    	m.nop
   1b4ca: 65 29 87 01  	l.andix r5, r5, 0x7
   1b4ce: 01 00        	c.nop
   1b4d0: 00 00 00 00  	v.nop
   1b4d4: 00 00 00 00 00 00    	m.nop
   1b4da: 31 15 c5 12  	l.st.w r5, [sp, 0x134]
   1b4de: 01 00        	c.nop
   1b4e0: 00 00 00 00  	v.nop
   1b4e4: 00 00 00 00 00 00    	m.nop
   1b4ea: 71 11 50 02  	l.ld.w r5, [sp, 0x30]
   1b4ee: 01 00        	c.nop
   1b4f0: 00 00 00 00  	v.nop
   1b4f4: 00 00 00 00 00 00    	m.nop
   1b4fa: 25 a1 97 01  	l.andix r4, r20, 0x17
   1b4fe: 01 00        	c.nop
   1b500: 00 00 00 00  	v.nop
   1b504: 00 00 00 00 00 00    	m.nop
   1b50a: 31 11 c4 10  	l.st.w r4, [sp, 0x104]
   1b50e: 01 00        	c.nop
   1b510: 00 00 00 00  	v.nop
   1b514: 00 00 00 00 00 00    	m.nop
   1b51a: 23 a1 95 18  	<unknown>
   1b51e: 01 00        	c.nop
   1b520: 00 00 00 00  	v.nop
   1b524: 00 00 00 00 00 00    	m.nop
   1b52a: 31 14 c4 18  	l.st.w r4, [sp, 0x190]
   1b52e: 01 00        	c.nop
   1b530: 00 00 00 00  	v.nop
   1b534: 00 00 00 00 00 00    	m.nop
   1b53a: 65 29 97 01  	l.andix r5, r5, 0x17
   1b53e: 01 00        	c.nop
   1b540: 00 00 00 00  	v.nop
   1b544: 00 00 00 00 00 00    	m.nop
   1b54a: 31 12 c5 10  	l.st.w r5, [sp, 0x108]
   1b54e: 01 00        	c.nop
   1b550: 00 00 00 00  	v.nop
   1b554: 00 00 00 00 00 00    	m.nop
   1b55a: 71 11 40 02  	l.ld.w r5, [sp, 0x20]
   1b55e: 01 00        	c.nop
   1b560: 00 00 00 00  	v.nop
   1b564: 00 00 00 00 00 00    	m.nop
   1b56a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b56e: 01 00        	c.nop
   1b570: 00 00 00 00  	v.nop
   1b574: 00 00 00 00 00 00    	m.nop
   1b57a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b57e: 01 00        	c.nop
   1b580: 00 00 00 00  	v.nop
   1b584: 00 00 00 00 00 00    	m.nop
   1b58a: 31 13 c4 0e  	l.st.w r4, [sp, 0xec]
   1b58e: 71 42        	<unknown>
   1b590: 00 00 00 00  	v.nop
   1b594: 00 00 00 00 00 00    	m.nop
   1b59a: 65 29 97 01  	l.andix r5, r5, 0x17
   1b59e: 01 00        	c.nop
   1b5a0: 00 00 00 00  	v.nop
   1b5a4: 00 00 00 00 00 00    	m.nop
   1b5aa: 31 13 c5 08  	l.st.w r5, [sp, 0x8c]
   1b5ae: 01 00        	c.nop
   1b5b0: 00 00 00 00  	v.nop
   1b5b4: 00 00 00 00 00 00    	m.nop
   1b5ba: 71 11 50 04  	l.ld.w r5, [sp, 0x50]
   1b5be: 01 00        	c.nop
   1b5c0: 00 00 00 00  	v.nop
   1b5c4: 00 00 00 00 00 00    	m.nop
   1b5ca: 65 29 87 01  	l.andix r5, r5, 0x7
   1b5ce: 01 00        	c.nop
   1b5d0: 00 00 00 00  	v.nop
   1b5d4: 00 00 00 00 00 00    	m.nop
   1b5da: 31 14 c5 12  	l.st.w r5, [sp, 0x130]
   1b5de: 01 00        	c.nop
   1b5e0: 00 00 00 00  	v.nop
   1b5e4: 00 00 00 00 00 00    	m.nop
   1b5ea: 71 11 48 0a  	l.ld.w r5, [sp, 0xa8]
   1b5ee: 01 00        	c.nop
   1b5f0: 00 00 00 00  	v.nop
   1b5f4: 00 00 00 00 00 00    	m.nop
   1b5fa: 23 29 04 10  	<unknown>
   1b5fe: 01 00        	c.nop
   1b600: 00 00 00 00  	v.nop
   1b604: 00 00 00 00 00 00    	m.nop
   1b60a: 31 16 c4 1a  	l.st.w r4, [sp, 0x1b8]
   1b60e: 01 00        	c.nop
   1b610: 00 00 00 00  	v.nop
   1b614: 00 00 00 00 00 00    	m.nop
   1b61a: 2f 20 70 1b  	l.ldi16.u r4, 0xde00
   1b61e: 01 00        	c.nop
   1b620: 00 00 00 00  	v.nop
   1b624: 00 00 00 00 00 00    	m.nop
   1b62a: 6f 22 80 00  	l.ldi_hi r4, 0x9
   1b62e: 01 00        	c.nop
   1b630: 00 00 00 00  	v.nop
   1b634: 00 00 00 00 00 00    	m.nop
   1b63a: 23 49 04 18  	<unknown>
   1b63e: 01 00        	c.nop
   1b640: 00 00 00 00  	v.nop
   1b644: 00 00 00 00 00 00    	m.nop
   1b64a: 31 15 c4 1a  	l.st.w r4, [sp, 0x1b4]
   1b64e: 51 42        	<unknown>
   1b650: 00 00 00 00  	v.nop
   1b654: 00 00 00 00 00 00    	m.nop
   1b65a: 23 29 04 10  	<unknown>
   1b65e: 01 00        	c.nop
   1b660: 00 00 00 00  	v.nop
   1b664: 00 00 00 00 00 00    	m.nop
   1b66a: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   1b66e: 01 00        	c.nop
   1b670: 00 00 00 00  	v.nop
   1b674: 00 00 00 00 00 00    	m.nop
   1b67a: 31 14 c4 1a  	l.st.w r4, [sp, 0x1b0]
   1b67e: 01 00        	c.nop
   1b680: 00 00 00 00  	v.nop
   1b684: 00 00 00 00 00 00    	m.nop
   1b68a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b68e: 01 00        	c.nop
   1b690: 00 00 00 00  	v.nop
   1b694: 00 00 00 00 00 00    	m.nop
   1b69a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b69e: 01 00        	c.nop
   1b6a0: 00 00 00 00  	v.nop
   1b6a4: 00 00 00 00 00 00    	m.nop
   1b6aa: 2f 2c 80 00  	l.ldi_hi r5, 0x10
   1b6ae: 01 00        	c.nop
   1b6b0: 00 00 00 00  	v.nop
   1b6b4: 00 00 00 00 00 00    	m.nop
   1b6ba: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   1b6be: 16 92        	<unknown>
   1b6c0: 00 00 00 00  	v.nop
   1b6c4: 00 00 00 00 00 00    	m.nop
   1b6ca: 31 15 c4 0c  	l.st.w r4, [sp, 0xd4]
   1b6ce: 01 00        	c.nop
   1b6d0: 00 00 00 00  	v.nop
   1b6d4: 00 00 00 00 00 00    	m.nop
   1b6da: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b6de: 01 00        	c.nop
   1b6e0: 00 00 00 00  	v.nop
   1b6e4: 00 00 00 00 00 00    	m.nop
   1b6ea: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b6ee: 01 00        	c.nop
   1b6f0: 00 00 00 00  	v.nop
   1b6f4: 00 00 00 00 00 00    	m.nop
   1b6fa: af 2b 80 00  	l.ldi_hi r5, 0xe
   1b6fe: 01 00        	c.nop
   1b700: 00 00 00 00  	v.nop
   1b704: 00 00 00 00 00 00    	m.nop
   1b70a: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   1b70e: 16 92        	<unknown>
   1b710: 00 00 00 00  	v.nop
   1b714: 00 00 00 00 00 00    	m.nop
   1b71a: 31 13 c4 12  	l.st.w r4, [sp, 0x12c]
   1b71e: 01 00        	c.nop
   1b720: 00 00 00 00  	v.nop
   1b724: 00 00 00 00 00 00    	m.nop
   1b72a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b72e: 01 00        	c.nop
   1b730: 00 00 00 00  	v.nop
   1b734: 00 00 00 00 00 00    	m.nop
   1b73a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b73e: 01 00        	c.nop
   1b740: 00 00 00 00  	v.nop
   1b744: 00 00 00 00 00 00    	m.nop
   1b74a: 2f 2b 80 00  	l.ldi_hi r5, 0xc
   1b74e: 01 00        	c.nop
   1b750: 00 00 00 00  	v.nop
   1b754: 00 00 00 00 00 00    	m.nop
   1b75a: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   1b75e: 16 92        	<unknown>
   1b760: 00 00 00 00  	v.nop
   1b764: 00 00 00 00 00 00    	m.nop
   1b76a: 31 14 c4 0c  	l.st.w r4, [sp, 0xd0]
   1b76e: 01 00        	c.nop
   1b770: 00 00 00 00  	v.nop
   1b774: 00 00 00 00 00 00    	m.nop
   1b77a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b77e: 01 00        	c.nop
   1b780: 00 00 00 00  	v.nop
   1b784: 00 00 00 00 00 00    	m.nop
   1b78a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b78e: 01 00        	c.nop
   1b790: 00 00 00 00  	v.nop
   1b794: 00 00 00 00 00 00    	m.nop
   1b79a: af 2a 80 00  	l.ldi_hi r5, 0xa
   1b79e: 01 00        	c.nop
   1b7a0: 00 00 00 00  	v.nop
   1b7a4: 00 00 00 00 00 00    	m.nop
   1b7aa: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   1b7ae: 16 92        	<unknown>
   1b7b0: 00 00 00 00  	v.nop
   1b7b4: 00 00 00 00 00 00    	m.nop
   1b7ba: 31 12 c4 12  	l.st.w r4, [sp, 0x128]
   1b7be: 01 00        	c.nop
   1b7c0: 00 00 00 00  	v.nop
   1b7c4: 00 00 00 00 00 00    	m.nop
   1b7ca: 2f 26 03 21  	l.ldi16.s r4, 0x78
   1b7ce: 01 00        	c.nop
   1b7d0: 00 00 00 00  	v.nop
   1b7d4: 00 00 00 00 00 00    	m.nop
   1b7da: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   1b7de: 01 00        	c.nop
   1b7e0: 00 00 00 00  	v.nop
   1b7e4: 00 00 00 00 00 00    	m.nop
   1b7ea: 23 29 04 10  	<unknown>
   1b7ee: 01 00        	c.nop
   1b7f0: 00 00 00 00  	v.nop
   1b7f4: 00 00 00 00 00 00    	m.nop
   1b7fa: 31 10 c4 08  	l.st.w r4, [sp, 0x80]
   1b7fe: 01 00        	c.nop
   1b800: 00 00 00 00  	v.nop
   1b804: 00 00 00 00 00 00    	m.nop
   1b80a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b80e: 01 00        	c.nop
   1b810: 00 00 00 00  	v.nop
   1b814: 00 00 00 00 00 00    	m.nop
   1b81a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b81e: 01 00        	c.nop
   1b820: 00 00 00 00  	v.nop
   1b824: 00 00 00 00 00 00    	m.nop
   1b82a: 31 11 c4 16  	l.st.w r4, [sp, 0x164]
   1b82e: 01 00        	c.nop
   1b830: 00 00 00 00  	v.nop
   1b834: 00 00 00 00 00 00    	m.nop
   1b83a: 2f 20 70 2b  	l.ldi16.s r4, 0x5e00
   1b83e: 01 00        	c.nop
   1b840: 00 00 00 00  	v.nop
   1b844: 00 00 00 00 00 00    	m.nop
   1b84a: af 24 81 00  	l.ldi_hi r4, 0x32
   1b84e: 01 00        	c.nop
   1b850: 00 00 00 00  	v.nop
   1b854: 00 00 00 00 00 00    	m.nop
   1b85a: 23 41 04 18  	<unknown>
   1b85e: 01 00        	c.nop
   1b860: 00 00 00 00  	v.nop
   1b864: 00 00 00 00 00 00    	m.nop
   1b86a: 23 21 87 0b  	l.srai r4, r4, 0x7
   1b86e: 01 00        	c.nop
   1b870: 00 00 00 00  	v.nop
   1b874: 00 00 00 00 00 00    	m.nop
   1b87a: 31 12 c4 22  	l.st.w r4, [sp, 0x228]
   1b87e: 01 00        	c.nop
   1b880: 00 00 00 00  	v.nop
   1b884: 00 00 00 00 00 00    	m.nop
   1b88a: 2f 24 40 11  	l.ldi16.u r4, 0x8810
   1b88e: 01 00        	c.nop
   1b890: 00 00 00 00  	v.nop
   1b894: 00 00 00 00 00 00    	m.nop
   1b89a: af 20 80 00  	l.ldi_hi r4, 0x2
   1b89e: 01 00        	c.nop
   1b8a0: 00 00 00 00  	v.nop
   1b8a4: 00 00 00 00 00 00    	m.nop
   1b8aa: 31 16 c4 14  	l.st.w r4, [sp, 0x158]
   1b8ae: 01 00        	c.nop
   1b8b0: 00 00 00 00  	v.nop
   1b8b4: 00 00 00 00 00 00    	m.nop
   1b8ba: 2f 20 22 13  	l.ldi16.u r4, 0x9440
   1b8be: 01 00        	c.nop
   1b8c0: 00 00 00 00  	v.nop
   1b8c4: 00 00 00 00 00 00    	m.nop
   1b8ca: af 20 80 00  	l.ldi_hi r4, 0x2
   1b8ce: 01 00        	c.nop
   1b8d0: 00 00 00 00  	v.nop
   1b8d4: 00 00 00 00 00 00    	m.nop
   1b8da: 2f 28 70 2f  	l.ldi16.s r5, 0x7e00
   1b8de: 01 00        	c.nop
   1b8e0: 00 00 00 00  	v.nop
   1b8e4: 00 00 00 00 00 00    	m.nop
   1b8ea: 31 13 c4 20  	l.st.w r4, [sp, 0x20c]
   1b8ee: 01 00        	c.nop
   1b8f0: 00 00 00 00  	v.nop
   1b8f4: 00 00 00 00 00 00    	m.nop
   1b8fa: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1b8fe: 01 00        	c.nop
   1b900: 00 00 00 00  	v.nop
   1b904: 00 00 00 00 00 00    	m.nop
   1b90a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b90e: 01 00        	c.nop
   1b910: 00 00 00 00  	v.nop
   1b914: 00 00 00 00 00 00    	m.nop
   1b91a: ef 2c 80 00  	l.ldi_hi r5, 0x13
   1b91e: 01 00        	c.nop
   1b920: 00 00 00 00  	v.nop
   1b924: 00 00 00 00 00 00    	m.nop
   1b92a: 00 00        	s.nop16
   1b92c: 00 00        	s.nop16
   1b92e: 16 92        	<unknown>
   1b930: 00 00 00 00  	v.nop
   1b934: 00 00 00 00 00 00    	m.nop
   1b93a: 31 13 c4 0c  	l.st.w r4, [sp, 0xcc]
   1b93e: 01 00        	c.nop
   1b940: 00 00 00 00  	v.nop
   1b944: 00 00 00 00 00 00    	m.nop
   1b94a: 2f 24 4a 05  	l.ldi16.u r4, 0x2950
   1b94e: 01 00        	c.nop
   1b950: 00 00 00 00  	v.nop
   1b954: 00 00 00 00 00 00    	m.nop
   1b95a: 6f 20 80 00  	l.ldi_hi r4, 0x1
   1b95e: 01 00        	c.nop
   1b960: 00 00 00 00  	v.nop
   1b964: 00 00 00 00 00 00    	m.nop
   1b96a: 31 15 c4 1e  	l.st.w r4, [sp, 0x1f4]
   1b96e: 01 00        	c.nop
   1b970: 00 00 00 00  	v.nop
   1b974: 00 00 00 00 00 00    	m.nop
   1b97a: 2f 24 58 03  	l.ldi16.u r4, 0x1b10
   1b97e: 01 00        	c.nop
   1b980: 00 00 00 00  	v.nop
   1b984: 00 00 00 00 00 00    	m.nop
   1b98a: 6f 20 80 00  	l.ldi_hi r4, 0x1
   1b98e: 01 00        	c.nop
   1b990: 00 00 00 00  	v.nop
   1b994: 00 00 00 00 00 00    	m.nop
   1b99a: 31 14 c4 1e  	l.st.w r4, [sp, 0x1f0]
   1b99e: 01 00        	c.nop
   1b9a0: 00 00 00 00  	v.nop
   1b9a4: 00 00 00 00 00 00    	m.nop
   1b9aa: 2f 24 2a 15  	l.ldi16.u r4, 0xa550
   1b9ae: 01 00        	c.nop
   1b9b0: 00 00 00 00  	v.nop
   1b9b4: 00 00 00 00 00 00    	m.nop
   1b9ba: af 20 80 00  	l.ldi_hi r4, 0x2
   1b9be: 01 00        	c.nop
   1b9c0: 00 00 00 00  	v.nop
   1b9c4: 00 00 00 00 00 00    	m.nop
   1b9ca: 31 17 c4 20  	l.st.w r4, [sp, 0x21c]
   1b9ce: 01 00        	c.nop
   1b9d0: 00 00 00 00  	v.nop
   1b9d4: 00 00 00 00 00 00    	m.nop
   1b9da: 2f 20 6b 09  	l.ldi16.u r4, 0x4d60
   1b9de: 01 00        	c.nop
   1b9e0: 00 00 00 00  	v.nop
   1b9e4: 00 00 00 00 00 00    	m.nop
   1b9ea: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1b9ee: 01 00        	c.nop
   1b9f0: 00 00 00 00  	v.nop
   1b9f4: 00 00 00 00 00 00    	m.nop
   1b9fa: 31 12 c4 20  	l.st.w r4, [sp, 0x208]
   1b9fe: 01 00        	c.nop
   1ba00: 00 00 00 00  	v.nop
   1ba04: 00 00 00 00 00 00    	m.nop
   1ba0a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1ba0e: 01 00        	c.nop
   1ba10: 00 00 00 00  	v.nop
   1ba14: 00 00 00 00 00 00    	m.nop
   1ba1a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1ba1e: 01 00        	c.nop
   1ba20: 00 00 00 00  	v.nop
   1ba24: 00 00 00 00 00 00    	m.nop
   1ba2a: 31 11 c4 20  	l.st.w r4, [sp, 0x204]
   1ba2e: 01 00        	c.nop
   1ba30: 00 00 00 00  	v.nop
   1ba34: 00 00 00 00 00 00    	m.nop
   1ba3a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1ba3e: 01 00        	c.nop
   1ba40: 00 00 00 00  	v.nop
   1ba44: 00 00 00 00 00 00    	m.nop
   1ba4a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1ba4e: 01 00        	c.nop
   1ba50: 00 00 00 00  	v.nop
   1ba54: 00 00 00 00 00 00    	m.nop
   1ba5a: 31 15 c4 14  	l.st.w r4, [sp, 0x154]
   1ba5e: 01 00        	c.nop
   1ba60: 00 00 00 00  	v.nop
   1ba64: 00 00 00 00 00 00    	m.nop
   1ba6a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1ba6e: 01 00        	c.nop
   1ba70: 00 00 00 00  	v.nop
   1ba74: 00 00 00 00 00 00    	m.nop
   1ba7a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1ba7e: 01 00        	c.nop
   1ba80: 00 00 00 00  	v.nop
   1ba84: 00 00 00 00 00 00    	m.nop
   1ba8a: 31 10 c4 20  	l.st.w r4, [sp, 0x200]
   1ba8e: 01 00        	c.nop
   1ba90: 00 00 00 00  	v.nop
   1ba94: 00 00 00 00 00 00    	m.nop
   1ba9a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1ba9e: 01 00        	c.nop
   1baa0: 00 00 00 00  	v.nop
   1baa4: 00 00 00 00 00 00    	m.nop
   1baaa: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1baae: 01 00        	c.nop
   1bab0: 00 00 00 00  	v.nop
   1bab4: 00 00 00 00 00 00    	m.nop
   1baba: 31 14 c4 14  	l.st.w r4, [sp, 0x150]
   1babe: 01 00        	c.nop
   1bac0: 00 00 00 00  	v.nop
   1bac4: 00 00 00 00 00 00    	m.nop
   1baca: 2f 24 37 17  	l.ldi16.u r4, 0xb6f0
   1bace: 01 00        	c.nop
   1bad0: 00 00 00 00  	v.nop
   1bad4: 00 00 00 00 00 00    	m.nop
   1bada: af 20 80 00  	l.ldi_hi r4, 0x2
   1bade: 01 00        	c.nop
   1bae0: 00 00 00 00  	v.nop
   1bae4: 00 00 00 00 00 00    	m.nop
   1baea: 31 11 c4 22  	l.st.w r4, [sp, 0x224]
   1baee: 01 00        	c.nop
   1baf0: 00 00 00 00  	v.nop
   1baf4: 00 00 00 00 00 00    	m.nop
   1bafa: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1bafe: 01 00        	c.nop
   1bb00: 00 00 00 00  	v.nop
   1bb04: 00 00 00 00 00 00    	m.nop
   1bb0a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1bb0e: 01 00        	c.nop
   1bb10: 00 00 00 00  	v.nop
   1bb14: 00 00 00 00 00 00    	m.nop
   1bb1a: 31 17 c4 0a  	l.st.w r4, [sp, 0xbc]
   1bb1e: 01 00        	c.nop
   1bb20: 00 00 00 00  	v.nop
   1bb24: 00 00 00 00 00 00    	m.nop
   1bb2a: 2f 20 10 01  	l.ldi16.u r4, 0x200
   1bb2e: 01 00        	c.nop
   1bb30: 00 00 00 00  	v.nop
   1bb34: 00 00 00 00 00 00    	m.nop
   1bb3a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1bb3e: 01 00        	c.nop
   1bb40: 00 00 00 00  	v.nop
   1bb44: 00 00 00 00 00 00    	m.nop
   1bb4a: 31 16 c4 0a  	l.st.w r4, [sp, 0xb8]
   1bb4e: 01 00        	c.nop
   1bb50: 00 00 00 00  	v.nop
   1bb54: 00 00 00 00 00 00    	m.nop
   1bb5a: 2f 20 54 1d  	l.ldi16.u r4, 0xea80
   1bb5e: 01 00        	c.nop
   1bb60: 00 00 00 00  	v.nop
   1bb64: 00 00 00 00 00 00    	m.nop
   1bb6a: af 20 80 00  	l.ldi_hi r4, 0x2
   1bb6e: 01 00        	c.nop
   1bb70: 00 00 00 00  	v.nop
   1bb74: 00 00 00 00 00 00    	m.nop
   1bb7a: 31 10 c4 22  	l.st.w r4, [sp, 0x220]
   1bb7e: 01 00        	c.nop
   1bb80: 00 00 00 00  	v.nop
   1bb84: 00 00 00 00 00 00    	m.nop
   1bb8a: 2f 24 1c 07  	l.ldi16.u r4, 0x3390
   1bb8e: 01 00        	c.nop
   1bb90: 00 00 00 00  	v.nop
   1bb94: 00 00 00 00 00 00    	m.nop
   1bb9a: ef 20 80 00  	l.ldi_hi r4, 0x3
   1bb9e: 01 00        	c.nop
   1bba0: 00 00 00 00  	v.nop
   1bba4: 00 00 00 00 00 00    	m.nop
   1bbaa: 31 14 c4 20  	l.st.w r4, [sp, 0x210]
   1bbae: 01 00        	c.nop
   1bbb0: 00 00 00 00  	v.nop
   1bbb4: 00 00 00 00 00 00    	m.nop
   1bbba: 2f 24 7d 0d  	l.ldi16.u r4, 0x6fb0
   1bbbe: 01 00        	c.nop
   1bbc0: 00 00 00 00  	v.nop
   1bbc4: 00 00 00 00 00 00    	m.nop
   1bbca: 2f 20 80 00  	l.ldi_hi r4, 0x0
   1bbce: 01 00        	c.nop
   1bbd0: 00 00 00 00  	v.nop
   1bbd4: 00 00 00 00 00 00    	m.nop
   1bbda: 31 17 c4 1e  	l.st.w r4, [sp, 0x1fc]
   1bbde: 01 00        	c.nop
   1bbe0: 00 00 00 00  	v.nop
   1bbe4: 00 00 00 00 00 00    	m.nop
   1bbea: 2b 30 80 01  	l.jal r0, 0x6 <0x1bc40>
   1bbee: 01 00        	c.nop
   1bbf0: 00 00 00 00  	v.nop
   1bbf4: 00 00 00 00 00 00    	m.nop
   1bbfa: 71 13 44 06  	l.ld.w r13, [sp, 0x64]
   1bbfe: 01 00        	c.nop
   1bc00: 00 00 00 00  	v.nop
   1bc04: 00 00 00 00 00 00    	m.nop
   1bc0a: 31 11 48 1a  	l.ld.w r4, [sp, 0x1a8]
   1bc0e: 01 00        	c.nop
   1bc10: 00 00 00 00  	v.nop
   1bc14: 00 00 00 00 00 00    	m.nop
   1bc1a: 31 11 44 1a  	l.ld.w r4, [sp, 0x1a4]
   1bc1e: 92 96        	<unknown>
   1bc20: 00 00 00 00  	v.nop
   1bc24: 00 00 00 00 00 00    	m.nop
   1bc2a: bb 68 84 c0  	l.blt.s r13, r4, 0x2 <0x1bc40>
   1bc2e: 01 00        	c.nop
   1bc30: 00 00 00 00  	v.nop
   1bc34: 00 00 00 00 00 00    	m.nop
   1bc3a: 2b 48 b9 03  	l.jal r0, 0xb99 <0x275c0>
   1bc3e: 01 00        	c.nop
   1bc40: 00 00 00 00  	v.nop
   1bc44: 00 00 00 00 00 00    	m.nop
   1bc4a: 31 11 5c 02  	l.ld.w r4, [sp, 0x3c]
   1bc4e: 01 00        	c.nop
   1bc50: 00 00 00 00  	v.nop
   1bc54: 00 00 00 00 00 00    	m.nop
   1bc5a: 71 11 50 0e  	l.ld.w r5, [sp, 0xf0]
   1bc5e: 01 00        	c.nop
   1bc60: 00 00 00 00  	v.nop
   1bc64: 00 00 00 00 00 00    	m.nop
   1bc6a: 31 11 cd 06  	l.st.w r13, [sp, 0x64]
   1bc6e: 01 00        	c.nop
   1bc70: 00 00 00 00  	v.nop
   1bc74: 00 00 00 00 00 00    	m.nop
   1bc7a: 25 69 04 1b  	<unknown>
   1bc7e: 01 00        	c.nop
   1bc80: 00 00 00 00  	v.nop
   1bc84: 00 00 00 00 00 00    	m.nop
   1bc8a: fb 20 05 00  	<unknown>
   1bc8e: 01 00        	c.nop
   1bc90: 00 00 00 00  	v.nop
   1bc94: 00 00 00 00 00 00    	m.nop
   1bc9a: f1 12 48 0e  	l.ld.w r11, [sp, 0xe8]
   1bc9e: 81 42        	<unknown>
   1bca0: 00 00 00 00  	v.nop
   1bca4: 00 00 00 00 00 00    	m.nop
   1bcaa: 2b 10 80 01  	l.jal r0, 0x2 <0x1bcc0>
   1bcae: 01 00        	c.nop
   1bcb0: 00 00 00 00  	v.nop
   1bcb4: 00 00 00 00 00 00    	m.nop
   1bcba: f1 12 48 0e  	l.ld.w r11, [sp, 0xe8]
   1bcbe: 85 42        	<unknown>
   1bcc0: 00 00 00 00  	v.nop
   1bcc4: 00 00 00 00 00 00    	m.nop
   1bcca: b1 11 44 0e  	l.ld.w r6, [sp, 0xe4]
   1bcce: 01 00        	c.nop
   1bcd0: 00 00 00 00  	v.nop
   1bcd4: 00 00 00 00 00 00    	m.nop
   1bcda: a5 29 06 02  	l.orix r6, r5, 0x6
   1bcde: 01 00        	c.nop
   1bce0: 00 00 00 00  	v.nop
   1bce4: 00 00 00 00 00 00    	m.nop
   1bcea: 71 11 5c 02  	l.ld.w r5, [sp, 0x3c]
   1bcee: 01 00        	c.nop
   1bcf0: 00 00 00 00  	v.nop
   1bcf4: 00 00 00 00 00 00    	m.nop
   1bcfa: 63 21 05 19  	<unknown>
   1bcfe: 01 00        	c.nop
   1bd00: 00 00 00 00  	v.nop
   1bd04: 00 00 00 00 00 00    	m.nop
   1bd0a: bb 35 20 40  	<unknown>
   1bd0e: 01 00        	c.nop
   1bd10: 00 00 00 00  	v.nop
   1bd14: 00 00 00 00 00 00    	m.nop
   1bd1a: 31 12 44 18  	l.ld.w r8, [sp, 0x184]
   1bd1e: 01 43        	<unknown>
   1bd20: 00 00 00 00  	v.nop
   1bd24: 00 00 00 00 00 00    	m.nop
   1bd2a: 00 00        	s.nop16
   1bd2c: 00 00        	s.nop16
   1bd2e: 96 83        	<unknown>
   1bd30: 00 00 00 00  	v.nop
   1bd34: 00 00 00 00 00 00    	m.nop
   1bd3a: bb 40 16 00  	<unknown>
   1bd3e: 01 00        	c.nop
   1bd40: 00 00 00 00  	v.nop
   1bd44: 00 00 00 00 00 00    	m.nop
   1bd4a: 00 00        	s.nop16
   1bd4c: 00 00        	s.nop16
   1bd4e: 9a 83        	<unknown>
   1bd50: 00 00 00 00  	v.nop
   1bd54: 00 00 00 00 00 00    	m.nop
   1bd5a: 23 3a 9f 0b  	l.srai r8, r7, 0x1f
   1bd5e: 01 00        	c.nop
   1bd60: 00 00 00 00  	v.nop
   1bd64: 00 00 00 00 00 00    	m.nop
   1bd6a: b1 12 48 18  	l.ld.w r10, [sp, 0x188]
   1bd6e: 0a 04        	<unknown>
   1bd70: 00 00 00 00  	v.nop
   1bd74: 00 00 00 00 00 00    	m.nop
   1bd7a: 63 3a 02 03  	l.slli r9, r7, 0x2
   1bd7e: 01 00        	c.nop
   1bd80: 00 00 00 00  	v.nop
   1bd84: 00 00 00 00 00 00    	m.nop
   1bd8a: e3 39 9e 03  	l.srli r7, r7, 0x1e
   1bd8e: 01 00        	c.nop
   1bd90: 00 00 00 00  	v.nop
   1bd94: 00 00 00 00 00 00    	m.nop
   1bd9a: e5 41 07 02  	l.orix r7, r8, 0x7
   1bd9e: 01 00        	c.nop
   1bda0: 00 00 00 00  	v.nop
   1bda4: 00 00 00 00 00 00    	m.nop
   1bdaa: 31 12 4c 18  	l.ld.w r8, [sp, 0x18c]
   1bdae: 01 00        	c.nop
   1bdb0: 00 00 00 00  	v.nop
   1bdb4: 00 00 00 00 00 00    	m.nop
   1bdba: 63 52 09 10  	<unknown>
   1bdbe: 01 00        	c.nop
   1bdc0: 00 00 00 00  	v.nop
   1bdc4: 00 00 00 00 00 00    	m.nop
   1bdca: 27 48 0a 00  	l.slt.u r9, r10
   1bdce: 01 00        	c.nop
   1bdd0: 00 00 00 00  	v.nop
   1bdd4: 00 00 00 00 00 00    	m.nop
   1bdda: ef 57 7f 1f  	l.ldi16.u r10, 0xffff
   1bdde: 01 00        	c.nop
   1bde0: 00 00 00 00  	v.nop
   1bde4: 00 00 00 00 00 00    	m.nop
   1bdea: e3 41 07 10  	<unknown>
   1bdee: 71 44        	<unknown>
   1bdf0: 00 00 00 00  	v.nop
   1bdf4: 00 00 00 00 00 00    	m.nop
   1bdfa: e1 39 01 40  	l.addi.u r7, r7, 0x1
   1bdfe: 01 00        	c.nop
   1be00: 00 00 00 00  	v.nop
   1be04: 00 00 00 00 00 00    	m.nop
   1be0a: 23 5a 08 10  	<unknown>
   1be0e: 01 00        	c.nop
   1be10: 00 00 00 00  	v.nop
   1be14: 00 00 00 00 00 00    	m.nop
   1be1a: 31 40 c9 00  	l.st.w r9, [r8]
   1be1e: 01 00        	c.nop
   1be20: 00 00 00 00  	v.nop
   1be24: 00 00 00 00 00 00    	m.nop
   1be2a: 71 12 4c 0e  	l.ld.w r9, [sp, 0xec]
   1be2e: 01 00        	c.nop
   1be30: 00 00 00 00  	v.nop
   1be34: 00 00 00 00 00 00    	m.nop
   1be3a: e5 39 8a 01  	l.andix r7, r7, 0xa
   1be3e: 01 00        	c.nop
   1be40: 00 00 00 00  	v.nop
   1be44: 00 00 00 00 00 00    	m.nop
   1be4a: 31 47 c7 fe  	l.st.w r7, [r8, -0x4]
   1be4e: 01 00        	c.nop
   1be50: 00 00 00 00  	v.nop
   1be54: 00 00 00 00 00 00    	m.nop
   1be5a: 61 4a 00 02  	l.addi.u r9, r9, 0x200
   1be5e: 01 00        	c.nop
   1be60: 00 00 00 00  	v.nop
   1be64: 00 00 00 00 00 00    	m.nop
   1be6a: 31 42 c9 00  	l.st.w r9, [r8, 0x8]
   1be6e: 01 00        	c.nop
   1be70: 00 00 00 00  	v.nop
   1be74: 00 00 00 00 00 00    	m.nop
   1be7a: 31 41 c6 00  	l.st.w r6, [r8, 0x4]
   1be7e: 01 00        	c.nop
   1be80: 00 00 00 00  	v.nop
   1be84: 00 00 00 00 00 00    	m.nop
   1be8a: f1 11 40 18  	l.ld.w r7, [sp, 0x180]
   1be8e: 01 00        	c.nop
   1be90: 00 00 00 00  	v.nop
   1be94: 00 00 00 00 00 00    	m.nop
   1be9a: 2f 32 10 21  	l.ldi16.s r6, 0x208
   1be9e: 01 00        	c.nop
   1bea0: 00 00 00 00  	v.nop
   1bea4: 00 00 00 00 00 00    	m.nop
   1beaa: 2f 30 81 00  	l.ldi_hi r6, 0x20
   1beae: 01 00        	c.nop
   1beb0: 00 00 00 00  	v.nop
   1beb4: 00 00 00 00 00 00    	m.nop
   1beba: a1 31 f9 2d  	l.addi.s r6, r6, -0x207
   1bebe: 01 00        	c.nop
   1bec0: 00 00 00 00  	v.nop
   1bec4: 00 00 00 00 00 00    	m.nop
   1beca: 31 45 c7 00  	l.st.w r7, [r8, 0x14]
   1bece: 91 43        	<unknown>
   1bed0: 00 00 00 00  	v.nop
   1bed4: 00 00 00 00 00 00    	m.nop
   1beda: 31 46 c6 fe  	l.st.w r6, [r8, -0x8]
   1bede: 01 00        	c.nop
   1bee0: 00 00 00 00  	v.nop
   1bee4: 00 00 00 00 00 00    	m.nop
   1beea: b3 b9 83 00  	l.movs r6, md23_wrk
   1beee: 01 00        	c.nop
   1bef0: 00 00 00 00  	v.nop
   1bef4: 00 00 00 00 00 00    	m.nop
   1befa: a5 31 87 01  	l.andix r6, r6, 0x7
   1befe: 01 00        	c.nop
   1bf00: 00 00 00 00  	v.nop
   1bf04: 00 00 00 00 00 00    	m.nop
   1bf0a: fb 33 00 00  	<unknown>
   1bf0e: 01 00        	c.nop
   1bf10: 00 00 00 00  	v.nop
   1bf14: 00 00 00 00 00 00    	m.nop
   1bf1a: 2f 32 04 21  	l.ldi16.s r6, 0x88
   1bf1e: 01 00        	c.nop
   1bf20: 00 00 00 00  	v.nop
   1bf24: 00 00 00 00 00 00    	m.nop
   1bf2a: a3 59 06 10  	<unknown>
   1bf2e: 01 00        	c.nop
   1bf30: 00 00 00 00  	v.nop
   1bf34: 00 00 00 00 00 00    	m.nop
   1bf3a: f1 31 40 00  	l.ld.w r7, [r6]
   1bf3e: 01 00        	c.nop
   1bf40: 00 00 00 00  	v.nop
   1bf44: 00 00 00 00 00 00    	m.nop
   1bf4a: 33 a2 83 00  	l.movs r8, md20_wrk
   1bf4e: 01 00        	c.nop
   1bf50: 00 00 00 00  	v.nop
   1bf54: 00 00 00 00 00 00    	m.nop
   1bf5a: 2f 50 f0 3e  	l.ldi_hi.m1 r10, 0xfe00
   1bf5e: 06 04        	<unknown>
   1bf60: 00 00 00 00  	v.nop
   1bf64: 00 00 00 00 00 00    	m.nop
   1bf6a: ef 57 7f 1f  	l.ldi16.u r10, 0xffff
   1bf6e: 69 8c        	<unknown>
   1bf70: 00 00 00 00  	v.nop
   1bf74: 00 00 00 00 00 00    	m.nop
   1bf7a: 73 02 80 00  	l.movs r9, sip_id
   1bf7e: 01 00        	c.nop
   1bf80: 00 00 00 00  	v.nop
   1bf84: 00 00 00 00 00 00    	m.nop
   1bf8a: 6f 50 80 00  	l.ldi_hi r10, 0x1
   1bf8e: be 04        	<unknown>
   1bf90: 00 00 00 00  	v.nop
   1bf94: 00 00 00 00 00 00    	m.nop
   1bf9a: e5 39 8a 01  	l.andix r7, r7, 0xa
   1bf9e: 01 00        	c.nop
   1bfa0: 00 00 00 00  	v.nop
   1bfa4: 00 00 00 00 00 00    	m.nop
   1bfaa: a1 ba 01 00  	l.addi.u r10, r23, 0x1
   1bfae: 01 00        	c.nop
   1bfb0: 00 00 00 00  	v.nop
   1bfb4: 00 00 00 00 00 00    	m.nop
   1bfba: e5 39 08 02  	l.orix r7, r7, 0x8
   1bfbe: e9 8c        	<unknown>
   1bfc0: 00 00 00 00  	v.nop
   1bfc4: 00 00 00 00 00 00    	m.nop
   1bfca: e5 39 09 02  	l.orix r7, r7, 0x9
   1bfce: 01 00        	c.nop
   1bfd0: 00 00 00 00  	v.nop
   1bfd4: 00 00 00 00 00 00    	m.nop
   1bfda: e5 39 1d 02  	l.orix r7, r7, 0x1d
   1bfde: 01 00        	c.nop
   1bfe0: 00 00 00 00  	v.nop
   1bfe4: 00 00 00 00 00 00    	m.nop
   1bfea: 31 30 c7 00  	l.st.w r7, [r6]
   1bfee: 01 00        	c.nop
   1bff0: 00 00 00 00  	v.nop
   1bff4: 00 00 00 00 00 00    	m.nop
   1bffa: 2f 35 01 21  	l.ldi16.s r6, 0x34
   1bffe: 01 00        	c.nop
   1c000: 00 00 00 00  	v.nop
   1c004: 00 00 00 00 00 00    	m.nop
   1c00a: 31 14 c4 0e  	l.st.w r4, [sp, 0xf0]
   1c00e: 01 00        	c.nop
   1c010: 00 00 00 00  	v.nop
   1c014: 00 00 00 00 00 00    	m.nop
   1c01a: a3 59 06 10  	<unknown>
   1c01e: 01 00        	c.nop
   1c020: 00 00 00 00  	v.nop
   1c024: 00 00 00 00 00 00    	m.nop
   1c02a: f3 01 80 00  	l.movs r7, sip_id
   1c02e: 01 00        	c.nop
   1c030: 00 00 00 00  	v.nop
   1c034: 00 00 00 00 00 00    	m.nop
   1c03a: 31 30 da 00  	l.st.w r26, [r6]
   1c03e: 01 00        	c.nop
   1c040: 00 00 00 00  	v.nop
   1c044: 00 00 00 00 00 00    	m.nop
   1c04a: b1 11 40 0e  	l.ld.w r6, [sp, 0xe0]
   1c04e: 01 00        	c.nop
   1c050: 00 00 00 00  	v.nop
   1c054: 00 00 00 00 00 00    	m.nop
   1c05a: 2d 30 20 00  	l.wait.dec r6
   1c05e: 01 00        	c.nop
   1c060: 00 00 00 00  	v.nop
   1c064: 00 00 00 00 00 00    	m.nop
   1c06a: 7b a6 ba ff  	l.blt.s r20, r26, -0x47 <0x1bbf0>
   1c06e: 01 00        	c.nop
   1c070: 00 00 00 00  	v.nop
   1c074: 00 00 00 00 00 00    	m.nop
   1c07a: b1 11 44 06  	l.ld.w r6, [sp, 0x64]
   1c07e: 81 49        	<unknown>
   1c080: 00 00 00 00  	v.nop
   1c084: 00 00 00 00 00 00    	m.nop
   1c08a: a3 22 15 19  	<unknown>
   1c08e: 01 00        	c.nop
   1c090: 00 00 00 00  	v.nop
   1c094: 00 00 00 00 00 00    	m.nop
   1c09a: e1 ba 01 00  	l.addi.u r11, r23, 0x1
   1c09e: 01 00        	c.nop
   1c0a0: 00 00 00 00  	v.nop
   1c0a4: 00 00 00 00 00 00    	m.nop
   1c0aa: 31 13 cb 02  	l.st.w r11, [sp, 0x2c]
   1c0ae: 01 00        	c.nop
   1c0b0: 00 00 00 00  	v.nop
   1c0b4: 00 00 00 00 00 00    	m.nop
   1c0ba: 23 52 9f 0b  	l.srai r8, r10, 0x1f
   1c0be: 01 00        	c.nop
   1c0c0: 00 00 00 00  	v.nop
   1c0c4: 00 00 00 00 00 00    	m.nop
   1c0ca: 63 31 85 18  	<unknown>
   1c0ce: 01 00        	c.nop
   1c0d0: 00 00 00 00  	v.nop
   1c0d4: 00 00 00 00 00 00    	m.nop
   1c0da: b1 11 58 18  	l.ld.w r6, [sp, 0x198]
   1c0de: 01 00        	c.nop
   1c0e0: 00 00 00 00  	v.nop
   1c0e4: 00 00 00 00 00 00    	m.nop
   1c0ea: 63 52 08 18  	<unknown>
   1c0ee: 01 00        	c.nop
   1c0f0: 00 00 00 00  	v.nop
   1c0f4: 00 00 00 00 00 00    	m.nop
   1c0fa: 25 4a 88 02  	l.xorix r8, r9, 0x8
   1c0fe: 01 00        	c.nop
   1c100: 00 00 00 00  	v.nop
   1c104: 00 00 00 00 00 00    	m.nop
   1c10a: 63 2a 87 03  	l.srli r9, r5, 0x7
   1c10e: 01 00        	c.nop
   1c110: 00 00 00 00  	v.nop
   1c114: 00 00 00 00 00 00    	m.nop
   1c11a: a3 53 9f 03  	l.srli r14, r10, 0x1f
   1c11e: ed 8c        	<unknown>
   1c120: 00 00 00 00  	v.nop
   1c124: 00 00 00 00 00 00    	m.nop
   1c12a: 65 2b 06 1b  	<unknown>
   1c12e: 01 00        	c.nop
   1c130: 00 00 00 00  	v.nop
   1c134: 00 00 00 00 00 00    	m.nop
   1c13a: a3 29 9f 0b  	l.srai r6, r5, 0x1f
   1c13e: 01 00        	c.nop
   1c140: 00 00 00 00  	v.nop
   1c144: 00 00 00 00 00 00    	m.nop
   1c14a: e3 29 06 18  	<unknown>
   1c14e: 01 00        	c.nop
   1c150: 00 00 00 00  	v.nop
   1c154: 00 00 00 00 00 00    	m.nop
   1c15a: a5 39 86 02  	l.xorix r6, r7, 0x6
   1c15e: 01 00        	c.nop
   1c160: 00 00 00 00  	v.nop
   1c164: 00 00 00 00 00 00    	m.nop
   1c16a: e3 21 14 19  	<unknown>
   1c16e: 01 00        	c.nop
   1c170: 00 00 00 00  	v.nop
   1c174: 00 00 00 00 00 00    	m.nop
   1c17a: 31 11 ca 08  	l.st.w r10, [sp, 0x84]
   1c17e: 01 00        	c.nop
   1c180: 00 00 00 00  	v.nop
   1c184: 00 00 00 00 00 00    	m.nop
   1c18a: a1 42 00 01  	l.addi.u r10, r8, 0x100
   1c18e: 01 00        	c.nop
   1c190: 00 00 00 00  	v.nop
   1c194: 00 00 00 00 00 00    	m.nop
   1c19a: 31 17 c7 0e  	l.st.w r7, [sp, 0xfc]
   1c19e: 01 00        	c.nop
   1c1a0: 00 00 00 00  	v.nop
   1c1a4: 00 00 00 00 00 00    	m.nop
   1c1aa: e5 31 97 01  	l.andix r7, r6, 0x17
   1c1ae: 01 00        	c.nop
   1c1b0: 00 00 00 00  	v.nop
   1c1b4: 00 00 00 00 00 00    	m.nop
   1c1ba: 25 42 97 01  	l.andix r8, r8, 0x17
   1c1be: 01 00        	c.nop
   1c1c0: 00 00 00 00  	v.nop
   1c1c4: 00 00 00 00 00 00    	m.nop
   1c1ca: 31 16 ce 10  	l.st.w r14, [sp, 0x118]
   1c1ce: 01 00        	c.nop
   1c1d0: 00 00 00 00  	v.nop
   1c1d4: 00 00 00 00 00 00    	m.nop
   1c1da: 31 13 d9 0a  	l.st.w r25, [sp, 0xac]
   1c1de: 01 00        	c.nop
   1c1e0: 00 00 00 00  	v.nop
   1c1e4: 00 00 00 00 00 00    	m.nop
   1c1ea: e3 6a 9f 0b  	l.srai r11, r13, 0x1f
   1c1ee: 01 00        	c.nop
   1c1f0: 00 00 00 00  	v.nop
   1c1f4: 00 00 00 00 00 00    	m.nop
   1c1fa: 23 6b 0b 18  	<unknown>
   1c1fe: 01 00        	c.nop
   1c200: 00 00 00 00  	v.nop
   1c204: 00 00 00 00 00 00    	m.nop
   1c20a: e3 6b 9f 03  	l.srli r15, r13, 0x1f
   1c20e: 2d 8e        	<unknown>
   1c210: 00 00 00 00  	v.nop
   1c214: 00 00 00 00 00 00    	m.nop
   1c21a: e1 62 01 00  	l.addi.u r11, r12, 0x1
   1c21e: 01 00        	c.nop
   1c220: 00 00 00 00  	v.nop
   1c224: 00 00 00 00 00 00    	m.nop
   1c22a: 31 16 cd 02  	l.st.w r13, [sp, 0x38]
   1c22e: 01 00        	c.nop
   1c230: 00 00 00 00  	v.nop
   1c234: 00 00 00 00 00 00    	m.nop
   1c23a: 65 5b 0a 02  	l.orix r13, r11, 0xa
   1c23e: 01 00        	c.nop
   1c240: 00 00 00 00  	v.nop
   1c244: 00 00 00 00 00 00    	m.nop
   1c24a: 31 12 cc 08  	l.st.w r12, [sp, 0x88]
   1c24e: 01 00        	c.nop
   1c250: 00 00 00 00  	v.nop
   1c254: 00 00 00 00 00 00    	m.nop
   1c25a: a5 62 97 01  	l.andix r10, r12, 0x17
   1c25e: 01 00        	c.nop
   1c260: 00 00 00 00  	v.nop
   1c264: 00 00 00 00 00 00    	m.nop
   1c26a: 23 73 18 03  	l.slli r12, r14, 0x18
   1c26e: 01 00        	c.nop
   1c270: 00 00 00 00  	v.nop
   1c274: 00 00 00 00 00 00    	m.nop
   1c27a: e3 7a 18 03  	l.slli r11, r15, 0x18
   1c27e: 01 00        	c.nop
   1c280: 00 00 00 00  	v.nop
   1c284: 00 00 00 00 00 00    	m.nop
   1c28a: 2f 70 88 20  	l.ldi_hi.m1 r14, 0x100
   1c28e: 01 00        	c.nop
   1c290: 00 00 00 00  	v.nop
   1c294: 00 00 00 00 00 00    	m.nop
   1c29a: 31 10 cf 04  	l.st.w r15, [sp, 0x40]
   1c29e: 01 00        	c.nop
   1c2a0: 00 00 00 00  	v.nop
   1c2a4: 00 00 00 00 00 00    	m.nop
   1c2aa: 7b 69 0e c0  	l.blt.u r13, r14, 0x5 <0x1c2f0>
   1c2ae: 01 00        	c.nop
   1c2b0: 00 00 00 00  	v.nop
   1c2b4: 00 00 00 00 00 00    	m.nop
   1c2ba: 2f 6a 10 21  	l.ldi16.s r13, 0x208
   1c2be: 01 00        	c.nop
   1c2c0: 00 00 00 00  	v.nop
   1c2c4: 00 00 00 00 00 00    	m.nop
   1c2ca: 2f 68 81 00  	l.ldi_hi r13, 0x20
   1c2ce: 01 00        	c.nop
   1c2d0: 00 00 00 00  	v.nop
   1c2d4: 00 00 00 00 00 00    	m.nop
   1c2da: 61 6b 0c 2f  	l.addi.s r13, r13, -0xf4
   1c2de: 01 00        	c.nop
   1c2e0: 00 00 00 00  	v.nop
   1c2e4: 00 00 00 00 00 00    	m.nop
   1c2ea: 31 13 cd 0a  	l.st.w r13, [sp, 0xac]
   1c2ee: 01 00        	c.nop
   1c2f0: 00 00 00 00  	v.nop
   1c2f4: 00 00 00 00 00 00    	m.nop
   1c2fa: 71 13 54 18  	l.ld.w r13, [sp, 0x194]
   1c2fe: 05 03        	<unknown>
   1c300: 00 00 00 00  	v.nop
   1c304: 00 00 00 00 00 00    	m.nop
   1c30a: b1 13 5c 0e  	l.ld.w r14, [sp, 0xfc]
   1c30e: 01 00        	c.nop
   1c310: 00 00 00 00  	v.nop
   1c314: 00 00 00 00 00 00    	m.nop
   1c31a: f1 13 4c 0e  	l.ld.w r15, [sp, 0xec]
   1c31e: 01 00        	c.nop
   1c320: 00 00 00 00  	v.nop
   1c324: 00 00 00 00 00 00    	m.nop
   1c32a: b1 14 5c 18  	l.ld.w r18, [sp, 0x19c]
   1c32e: 01 00        	c.nop
   1c330: 00 00 00 00  	v.nop
   1c334: 00 00 00 00 00 00    	m.nop
   1c33a: 63 6b 0e 19  	<unknown>
   1c33e: 01 00        	c.nop
   1c340: 00 00 00 00  	v.nop
   1c344: 00 00 00 00 00 00    	m.nop
   1c34a: b1 13 50 18  	l.ld.w r14, [sp, 0x190]
   1c34e: 01 00        	c.nop
   1c350: 00 00 00 00  	v.nop
   1c354: 00 00 00 00 00 00    	m.nop
   1c35a: 23 6c 01 03  	l.slli r16, r13, 0x1
   1c35e: 01 00        	c.nop
   1c360: 00 00 00 00  	v.nop
   1c364: 00 00 00 00 00 00    	m.nop
   1c36a: 23 94 10 10  	<unknown>
   1c36e: 01 00        	c.nop
   1c370: 00 00 00 00  	v.nop
   1c374: 00 00 00 00 00 00    	m.nop
   1c37a: 31 13 d0 1e  	l.st.w r16, [sp, 0x1ec]
   1c37e: 01 00        	c.nop
   1c380: 00 00 00 00  	v.nop
   1c384: 00 00 00 00 00 00    	m.nop
   1c38a: 31 15 ce 0a  	l.st.w r14, [sp, 0xb4]
   1c38e: 01 00        	c.nop
   1c390: 00 00 00 00  	v.nop
   1c394: 00 00 00 00 00 00    	m.nop
   1c39a: a3 2b 02 03  	l.slli r14, r5, 0x2
   1c39e: 01 00        	c.nop
   1c3a0: 00 00 00 00  	v.nop
   1c3a4: 00 00 00 00 00 00    	m.nop
   1c3aa: a3 7b 0e 18  	<unknown>
   1c3ae: 01 00        	c.nop
   1c3b0: 00 00 00 00  	v.nop
   1c3b4: 00 00 00 00 00 00    	m.nop
   1c3ba: e3 6b 9f 0b  	l.srai r15, r13, 0x1f
   1c3be: fd 82        	<unknown>
   1c3c0: 00 00 00 00  	v.nop
   1c3c4: 00 00 00 00 00 00    	m.nop
   1c3ca: 00 00        	s.nop16
   1c3cc: 00 00        	s.nop16
   1c3ce: 86 07        	<unknown>
   1c3d0: 00 00 00 00  	v.nop
   1c3d4: 00 00 00 00 00 00    	m.nop
   1c3da: 65 7b 0d 02  	l.orix r13, r15, 0xd
   1c3de: 01 00        	c.nop
   1c3e0: 00 00 00 00  	v.nop
   1c3e4: 00 00 00 00 00 00    	m.nop
   1c3ea: f1 13 40 1a  	l.ld.w r15, [sp, 0x1a0]
   1c3ee: 01 00        	c.nop
   1c3f0: 00 00 00 00  	v.nop
   1c3f4: 00 00 00 00 00 00    	m.nop
   1c3fa: 27 80 12 00  	l.slt.u r16, r18
   1c3fe: 01 00        	c.nop
   1c400: 00 00 00 00  	v.nop
   1c404: 00 00 00 00 00 00    	m.nop
   1c40a: 31 11 c6 1e  	l.st.w r6, [sp, 0x1e4]
   1c40e: 01 00        	c.nop
   1c410: 00 00 00 00  	v.nop
   1c414: 00 00 00 00 00 00    	m.nop
   1c41a: b1 11 58 02  	l.ld.w r6, [sp, 0x38]
   1c41e: 01 00        	c.nop
   1c420: 00 00 00 00  	v.nop
   1c424: 00 00 00 00 00 00    	m.nop
   1c42a: b1 14 54 08  	l.ld.w r18, [sp, 0x94]
   1c42e: 01 00        	c.nop
   1c430: 00 00 00 00  	v.nop
   1c434: 00 00 00 00 00 00    	m.nop
   1c43a: 63 7b 0d 10  	<unknown>
   1c43e: 81 47        	<unknown>
   1c440: 00 00 00 00  	v.nop
   1c444: 00 00 00 00 00 00    	m.nop
   1c44a: 61 6b 01 40  	l.addi.u r13, r13, 0x1
   1c44e: 01 00        	c.nop
   1c450: 00 00 00 00  	v.nop
   1c454: 00 00 00 00 00 00    	m.nop
   1c45a: e5 39 09 02  	l.orix r7, r7, 0x9
   1c45e: ce 84        	<unknown>
   1c460: 00 00 00 00  	v.nop
   1c464: 00 00 00 00 00 00    	m.nop
   1c46a: a3 49 86 18  	<unknown>
   1c46e: 01 00        	c.nop
   1c470: 00 00 00 00  	v.nop
   1c474: 00 00 00 00 00 00    	m.nop
   1c47a: 31 10 c6 10  	l.st.w r6, [sp, 0x100]
   1c47e: 01 00        	c.nop
   1c480: 00 00 00 00  	v.nop
   1c484: 00 00 00 00 00 00    	m.nop
   1c48a: a5 51 0b 02  	l.orix r6, r10, 0xb
   1c48e: 01 00        	c.nop
   1c490: 00 00 00 00  	v.nop
   1c494: 00 00 00 00 00 00    	m.nop
   1c49a: 31 15 c6 10  	l.st.w r6, [sp, 0x114]
   1c49e: 01 00        	c.nop
   1c4a0: 00 00 00 00  	v.nop
   1c4a4: 00 00 00 00 00 00    	m.nop
   1c4aa: a5 41 0c 02  	l.orix r6, r8, 0xc
   1c4ae: 01 00        	c.nop
   1c4b0: 00 00 00 00  	v.nop
   1c4b4: 00 00 00 00 00 00    	m.nop
   1c4ba: 31 14 c6 10  	l.st.w r6, [sp, 0x110]
   1c4be: 01 00        	c.nop
   1c4c0: 00 00 00 00  	v.nop
   1c4c4: 00 00 00 00 00 00    	m.nop
   1c4ca: a1 71 00 02  	l.addi.u r6, r14, 0x200
   1c4ce: 01 00        	c.nop
   1c4d0: 00 00 00 00  	v.nop
   1c4d4: 00 00 00 00 00 00    	m.nop
   1c4da: 31 17 c6 10  	l.st.w r6, [sp, 0x11c]
   1c4de: 01 00        	c.nop
   1c4e0: 00 00 00 00  	v.nop
   1c4e4: 00 00 00 00 00 00    	m.nop
   1c4ea: ef 37 7f 1f  	l.ldi16.u r6, 0xffff
   1c4ee: 01 00        	c.nop
   1c4f0: 00 00 00 00  	v.nop
   1c4f4: 00 00 00 00 00 00    	m.nop
   1c4fa: a5 69 86 01  	l.andix r6, r13, 0x6
   1c4fe: 01 00        	c.nop
   1c500: 00 00 00 00  	v.nop
   1c504: 00 00 00 00 00 00    	m.nop
   1c50a: 31 10 c6 1e  	l.st.w r6, [sp, 0x1e0]
   1c50e: 01 00        	c.nop
   1c510: 00 00 00 00  	v.nop
   1c514: 00 00 00 00 00 00    	m.nop
   1c51a: a5 29 97 01  	l.andix r6, r5, 0x17
   1c51e: 85 02        	<unknown>
   1c520: 00 00 00 00  	v.nop
   1c524: 00 00 00 00 00 00    	m.nop
   1c52a: 31 17 c5 1c  	l.st.w r5, [sp, 0x1dc]
   1c52e: 01 00        	c.nop
   1c530: 00 00 00 00  	v.nop
   1c534: 00 00 00 00 00 00    	m.nop
   1c53a: 65 21 97 01  	l.andix r5, r4, 0x17
   1c53e: 05 02        	<unknown>
   1c540: 00 00 00 00  	v.nop
   1c544: 00 00 00 00 00 00    	m.nop
   1c54a: 31 16 c5 1c  	l.st.w r5, [sp, 0x1d8]
   1c54e: 01 00        	c.nop
   1c550: 00 00 00 00  	v.nop
   1c554: 00 00 00 00 00 00    	m.nop
   1c55a: 71 11 44 08  	l.ld.w r5, [sp, 0x84]
   1c55e: 01 00        	c.nop
   1c560: 00 00 00 00  	v.nop
   1c564: 00 00 00 00 00 00    	m.nop
   1c56a: 31 15 c4 1c  	l.st.w r4, [sp, 0x1d4]
   1c56e: 01 00        	c.nop
   1c570: 00 00 00 00  	v.nop
   1c574: 00 00 00 00 00 00    	m.nop
   1c57a: 2f 20 78 3f  	l.ldi16.s r4, -0x100
   1c57e: 01 00        	c.nop
   1c580: 00 00 00 00  	v.nop
   1c584: 00 00 00 00 00 00    	m.nop
   1c58a: 31 12 c7 1e  	l.st.w r7, [sp, 0x1e8]
   1c58e: 01 00        	c.nop
   1c590: 00 00 00 00  	v.nop
   1c594: 00 00 00 00 00 00    	m.nop
   1c59a: 31 13 c6 10  	l.st.w r6, [sp, 0x10c]
   1c59e: 01 00        	c.nop
   1c5a0: 00 00 00 00  	v.nop
   1c5a4: 00 00 00 00 00 00    	m.nop
   1c5aa: 23 21 85 18  	<unknown>
   1c5ae: 01 00        	c.nop
   1c5b0: 00 00 00 00  	v.nop
   1c5b4: 00 00 00 00 00 00    	m.nop
   1c5ba: 31 13 c4 1a  	l.st.w r4, [sp, 0x1ac]
   1c5be: 05 42        	<unknown>
   1c5c0: 00 00 00 00  	v.nop
   1c5c4: 00 00 00 00 00 00    	m.nop
   1c5ca: 31 12 c4 04  	l.st.w r4, [sp, 0x48]
   1c5ce: 01 00        	c.nop
   1c5d0: 00 00 00 00  	v.nop
   1c5d4: 00 00 00 00 00 00    	m.nop
   1c5da: 2b 58 80 01  	l.jal r0, 0xb <0x1c680>
   1c5de: 01 00        	c.nop
   1c5e0: 00 00 00 00  	v.nop
   1c5e4: 00 00 00 00 00 00    	m.nop
   1c5ea: 71 12 50 06  	l.ld.w r9, [sp, 0x70]
   1c5ee: 01 00        	c.nop
   1c5f0: 00 00 00 00  	v.nop
   1c5f4: 00 00 00 00 00 00    	m.nop
   1c5fa: 33 01 80 00  	l.movs r4, sip_id
   1c5fe: 01 00        	c.nop
   1c600: 00 00 00 00  	v.nop
   1c604: 00 00 00 00 00 00    	m.nop
   1c60a: 31 38 da 00  	l.st.w r26, [r7]
   1c60e: 01 00        	c.nop
   1c610: 00 00 00 00  	v.nop
   1c614: 00 00 00 00 00 00    	m.nop
   1c61a: 31 11 4c 06  	l.ld.w r4, [sp, 0x6c]
   1c61e: 01 00        	c.nop
   1c620: 00 00 00 00  	v.nop
   1c624: 00 00 00 00 00 00    	m.nop
   1c62a: 61 4a 40 00  	l.addi.u r9, r9, 0x40
   1c62e: 01 00        	c.nop
   1c630: 00 00 00 00  	v.nop
   1c634: 00 00 00 00 00 00    	m.nop
   1c63a: 2d 20 20 00  	l.wait.dec r4
   1c63e: 01 00        	c.nop
   1c640: 00 00 00 00  	v.nop
   1c644: 00 00 00 00 00 00    	m.nop
   1c64a: 31 11 54 0a  	l.ld.w r4, [sp, 0xb4]
   1c64e: 01 00        	c.nop
   1c650: 00 00 00 00  	v.nop
   1c654: 00 00 00 00 00 00    	m.nop
   1c65a: 21 21 c0 2f  	l.addi.s r4, r4, -0x40
   1c65e: 01 00        	c.nop
   1c660: 00 00 00 00  	v.nop
   1c664: 00 00 00 00 00 00    	m.nop
   1c66a: 31 15 c4 0a  	l.st.w r4, [sp, 0xb4]
   1c66e: 01 00        	c.nop
   1c670: 00 00 00 00  	v.nop
   1c674: 00 00 00 00 00 00    	m.nop
   1c67a: 3b 4e d4 be  	l.bge.s r9, r20, -0xa8 <0x1bbf0>
   1c67e: 01 00        	c.nop
   1c680: 00 00 00 00  	v.nop
   1c684: 00 00 00 00 00 00    	m.nop
   1c68a: 31 11 5c 0e  	l.ld.w r4, [sp, 0xfc]
   1c68e: 01 00        	c.nop
   1c690: 00 00 00 00  	v.nop
   1c694: 00 00 00 00 00 00    	m.nop
   1c69a: 31 12 4c 02  	l.ld.w r8, [sp, 0x2c]
   1c69e: 01 00        	c.nop
   1c6a0: 00 00 00 00  	v.nop
   1c6a4: 00 00 00 00 00 00    	m.nop
   1c6aa: 31 14 c9 06  	l.st.w r9, [sp, 0x70]
   1c6ae: 01 00        	c.nop
   1c6b0: 00 00 00 00  	v.nop
   1c6b4: 00 00 00 00 00 00    	m.nop
   1c6ba: 63 49 04 18  	<unknown>
   1c6be: 66 82        	<unknown>
   1c6c0: 00 00 00 00  	v.nop
   1c6c4: 00 00 00 00 00 00    	m.nop
   1c6ca: a3 29 9f 0b  	l.srai r6, r5, 0x1f
   1c6ce: 01 00        	c.nop
   1c6d0: 00 00 00 00  	v.nop
   1c6d4: 00 00 00 00 00 00    	m.nop
   1c6da: e3 29 06 18  	<unknown>
   1c6de: 01 00        	c.nop
   1c6e0: 00 00 00 00  	v.nop
   1c6e4: 00 00 00 00 00 00    	m.nop
   1c6ea: 63 29 87 03  	l.srli r5, r5, 0x7
   1c6ee: 01 00        	c.nop
   1c6f0: 00 00 00 00  	v.nop
   1c6f4: 00 00 00 00 00 00    	m.nop
   1c6fa: 65 29 88 01  	l.andix r5, r5, 0x8
   1c6fe: 01 00        	c.nop
   1c700: 00 00 00 00  	v.nop
   1c704: 00 00 00 00 00 00    	m.nop
   1c70a: 31 12 44 1e  	l.ld.w r8, [sp, 0x1e4]
   1c70e: 01 00        	c.nop
   1c710: 00 00 00 00  	v.nop
   1c714: 00 00 00 00 00 00    	m.nop
   1c71a: a5 39 86 02  	l.xorix r6, r7, 0x6
   1c71e: 01 00        	c.nop
   1c720: 00 00 00 00  	v.nop
   1c724: 00 00 00 00 00 00    	m.nop
   1c72a: e1 31 40 00  	l.addi.u r7, r6, 0x40
   1c72e: 01 00        	c.nop
   1c730: 00 00 00 00  	v.nop
   1c734: 00 00 00 00 00 00    	m.nop
   1c73a: 71 12 58 1a  	l.ld.w r9, [sp, 0x1b8]
   1c73e: 01 00        	c.nop
   1c740: 00 00 00 00  	v.nop
   1c744: 00 00 00 00 00 00    	m.nop
   1c74a: a5 31 97 01  	l.andix r6, r6, 0x17
   1c74e: 01 00        	c.nop
   1c750: 00 00 00 00  	v.nop
   1c754: 00 00 00 00 00 00    	m.nop
   1c75a: e5 39 08 02  	l.orix r7, r7, 0x8
   1c75e: 01 00        	c.nop
   1c760: 00 00 00 00  	v.nop
   1c764: 00 00 00 00 00 00    	m.nop
   1c76a: 31 12 58 0e  	l.ld.w r8, [sp, 0xf8]
   1c76e: 01 00        	c.nop
   1c770: 00 00 00 00  	v.nop
   1c774: 00 00 00 00 00 00    	m.nop
   1c77a: 65 31 05 02  	l.orix r5, r6, 0x5
   1c77e: 01 00        	c.nop
   1c780: 00 00 00 00  	v.nop
   1c784: 00 00 00 00 00 00    	m.nop
   1c78a: 31 48 c8 00  	l.st.w r8, [r9]
   1c78e: 01 00        	c.nop
   1c790: 00 00 00 00  	v.nop
   1c794: 00 00 00 00 00 00    	m.nop
   1c79a: 31 12 48 1c  	l.ld.w r8, [sp, 0x1c8]
   1c79e: 01 00        	c.nop
   1c7a0: 00 00 00 00  	v.nop
   1c7a4: 00 00 00 00 00 00    	m.nop
   1c7aa: 31 4f c8 fe  	l.st.w r8, [r9, -0x4]
   1c7ae: 01 00        	c.nop
   1c7b0: 00 00 00 00  	v.nop
   1c7b4: 00 00 00 00 00 00    	m.nop
   1c7ba: 31 12 54 1a  	l.ld.w r8, [sp, 0x1b4]
   1c7be: 01 00        	c.nop
   1c7c0: 00 00 00 00  	v.nop
   1c7c4: 00 00 00 00 00 00    	m.nop
   1c7ca: 31 4a c8 00  	l.st.w r8, [r9, 0x8]
   1c7ce: 01 00        	c.nop
   1c7d0: 00 00 00 00  	v.nop
   1c7d4: 00 00 00 00 00 00    	m.nop
   1c7da: 31 49 cf 00  	l.st.w r15, [r9, 0x4]
   1c7de: 01 00        	c.nop
   1c7e0: 00 00 00 00  	v.nop
   1c7e4: 00 00 00 00 00 00    	m.nop
   1c7ea: 31 4a de 0a  	l.st.w r30, [r9, 0xa8]
   1c7ee: 01 00        	c.nop
   1c7f0: 00 00 00 00  	v.nop
   1c7f4: 00 00 00 00 00 00    	m.nop
   1c7fa: 31 12 50 0a  	l.ld.w r8, [sp, 0xb0]
   1c7fe: 01 00        	c.nop
   1c800: 00 00 00 00  	v.nop
   1c804: 00 00 00 00 00 00    	m.nop
   1c80a: 31 4b c8 0a  	l.st.w r8, [r9, 0xac]
   1c80e: 01 00        	c.nop
   1c810: 00 00 00 00  	v.nop
   1c814: 00 00 00 00 00 00    	m.nop
   1c81a: 31 12 4c 1c  	l.ld.w r8, [sp, 0x1cc]
   1c81e: 01 00        	c.nop
   1c820: 00 00 00 00  	v.nop
   1c824: 00 00 00 00 00 00    	m.nop
   1c82a: 31 4c c8 0a  	l.st.w r8, [r9, 0xb0]
   1c82e: 01 00        	c.nop
   1c830: 00 00 00 00  	v.nop
   1c834: 00 00 00 00 00 00    	m.nop
   1c83a: 31 12 44 1c  	l.ld.w r8, [sp, 0x1c4]
   1c83e: 01 00        	c.nop
   1c840: 00 00 00 00  	v.nop
   1c844: 00 00 00 00 00 00    	m.nop
   1c84a: 31 4d c8 0a  	l.st.w r8, [r9, 0xb4]
   1c84e: 01 00        	c.nop
   1c850: 00 00 00 00  	v.nop
   1c854: 00 00 00 00 00 00    	m.nop
   1c85a: 31 4c da 00  	l.st.w r26, [r9, 0x10]
   1c85e: 01 00        	c.nop
   1c860: 00 00 00 00  	v.nop
   1c864: 00 00 00 00 00 00    	m.nop
   1c86a: 31 49 cf 0c  	l.st.w r15, [r9, 0xc4]
   1c86e: 01 00        	c.nop
   1c870: 00 00 00 00  	v.nop
   1c874: 00 00 00 00 00 00    	m.nop
   1c87a: 31 12 48 1e  	l.ld.w r8, [sp, 0x1e8]
   1c87e: 01 00        	c.nop
   1c880: 00 00 00 00  	v.nop
   1c884: 00 00 00 00 00 00    	m.nop
   1c88a: 31 4a c8 0c  	l.st.w r8, [r9, 0xc8]
   1c88e: 01 00        	c.nop
   1c890: 00 00 00 00  	v.nop
   1c894: 00 00 00 00 00 00    	m.nop
   1c89a: 31 4b cf 0c  	l.st.w r15, [r9, 0xcc]
   1c89e: 01 00        	c.nop
   1c8a0: 00 00 00 00  	v.nop
   1c8a4: 00 00 00 00 00 00    	m.nop
   1c8aa: 31 49 c5 06  	l.st.w r5, [r9, 0x64]
   1c8ae: 01 00        	c.nop
   1c8b0: 00 00 00 00  	v.nop
   1c8b4: 00 00 00 00 00 00    	m.nop
   1c8ba: 2f 28 08 21  	l.ldi16.s r5, 0x100
   1c8be: 01 00        	c.nop
   1c8c0: 00 00 00 00  	v.nop
   1c8c4: 00 00 00 00 00 00    	m.nop
   1c8ca: 31 49 c5 0a  	l.st.w r5, [r9, 0xa4]
   1c8ce: 01 00        	c.nop
   1c8d0: 00 00 00 00  	v.nop
   1c8d4: 00 00 00 00 00 00    	m.nop
   1c8da: 2f 28 02 21  	l.ldi16.s r5, 0x40
   1c8de: 01 00        	c.nop
   1c8e0: 00 00 00 00  	v.nop
   1c8e4: 00 00 00 00 00 00    	m.nop
   1c8ea: 31 4e da 0a  	l.st.w r26, [r9, 0xb8]
   1c8ee: 01 00        	c.nop
   1c8f0: 00 00 00 00  	v.nop
   1c8f4: 00 00 00 00 00 00    	m.nop
   1c8fa: 31 4f da 0a  	l.st.w r26, [r9, 0xbc]
   1c8fe: 01 00        	c.nop
   1c900: 00 00 00 00  	v.nop
   1c904: 00 00 00 00 00 00    	m.nop
   1c90a: 31 48 c5 0c  	l.st.w r5, [r9, 0xc0]
   1c90e: 01 00        	c.nop
   1c910: 00 00 00 00  	v.nop
   1c914: 00 00 00 00 00 00    	m.nop
   1c91a: 2f 28 88 20  	l.ldi_hi.m1 r5, 0x100
   1c91e: 01 00        	c.nop
   1c920: 00 00 00 00  	v.nop
   1c924: 00 00 00 00 00 00    	m.nop
   1c92a: 31 4c cf 0c  	l.st.w r15, [r9, 0xd0]
   1c92e: 01 00        	c.nop
   1c930: 00 00 00 00  	v.nop
   1c934: 00 00 00 00 00 00    	m.nop
   1c93a: 3b 39 05 c0  	l.blt.u r7, r5, 0x4 <0x1c970>
   1c93e: 01 00        	c.nop
   1c940: 00 00 00 00  	v.nop
   1c944: 00 00 00 00 00 00    	m.nop
   1c94a: 2f 22 10 21  	l.ldi16.s r4, 0x208
   1c94e: 01 00        	c.nop
   1c950: 00 00 00 00  	v.nop
   1c954: 00 00 00 00 00 00    	m.nop
   1c95a: 2f 20 81 00  	l.ldi_hi r4, 0x20
   1c95e: 01 00        	c.nop
   1c960: 00 00 00 00  	v.nop
   1c964: 00 00 00 00 00 00    	m.nop
   1c96a: 21 21 0c 2f  	l.addi.s r4, r4, -0xf4
   1c96e: 01 00        	c.nop
   1c970: 00 00 00 00  	v.nop
   1c974: 00 00 00 00 00 00    	m.nop
   1c97a: 71 11 50 1a  	l.ld.w r5, [sp, 0x1b0]
   1c97e: 01 00        	c.nop
   1c980: 00 00 00 00  	v.nop
   1c984: 00 00 00 00 00 00    	m.nop
   1c98a: 31 28 c4 00  	l.st.w r4, [r5]
   1c98e: 91 42        	<unknown>
   1c990: 00 00 00 00  	v.nop
   1c994: 00 00 00 00 00 00    	m.nop
   1c99a: 33 b9 83 00  	l.movs r4, md23_wrk
   1c99e: 01 00        	c.nop
   1c9a0: 00 00 00 00  	v.nop
   1c9a4: 00 00 00 00 00 00    	m.nop
   1c9aa: 25 21 85 01  	l.andix r4, r4, 0x5
   1c9ae: 01 00        	c.nop
   1c9b0: 00 00 00 00  	v.nop
   1c9b4: 00 00 00 00 00 00    	m.nop
   1c9ba: bb 24 00 00  	<unknown>
   1c9be: 01 00        	c.nop
   1c9c0: 00 00 00 00  	v.nop
   1c9c4: 00 00 00 00 00 00    	m.nop
   1c9ca: 31 11 48 0a  	l.ld.w r4, [sp, 0xa8]
   1c9ce: 01 00        	c.nop
   1c9d0: 00 00 00 00  	v.nop
   1c9d4: 00 00 00 00 00 00    	m.nop
   1c9da: 2f 2a 04 21  	l.ldi16.s r5, 0x88
   1c9de: 01 00        	c.nop
   1c9e0: 00 00 00 00  	v.nop
   1c9e4: 00 00 00 00 00 00    	m.nop
   1c9ea: 2f 40 f0 3e  	l.ldi_hi.m1 r8, 0xfe00
   1c9ee: 01 00        	c.nop
   1c9f0: 00 00 00 00  	v.nop
   1c9f4: 00 00 00 00 00 00    	m.nop
   1c9fa: 00 00        	s.nop16
   1c9fc: 00 00        	s.nop16
   1c9fe: 16 92        	<unknown>
   1ca00: 00 00 00 00  	v.nop
   1ca04: 00 00 00 00 00 00    	m.nop
   1ca0a: 71 21 40 00  	l.ld.w r5, [r4]
   1ca0e: 01 00        	c.nop
   1ca10: 00 00 00 00  	v.nop
   1ca14: 00 00 00 00 00 00    	m.nop
   1ca1a: b3 a1 83 00  	l.movs r6, md20_wrk
   1ca1e: 01 00        	c.nop
   1ca20: 00 00 00 00  	v.nop
   1ca24: 00 00 00 00 00 00    	m.nop
   1ca2a: f3 01 80 00  	l.movs r7, sip_id
   1ca2e: 06 03        	<unknown>
   1ca30: 00 00 00 00  	v.nop
   1ca34: 00 00 00 00 00 00    	m.nop
   1ca3a: a5 31 88 01  	l.andix r6, r6, 0x8
   1ca3e: be 03        	<unknown>
   1ca40: 00 00 00 00  	v.nop
   1ca44: 00 00 00 00 00 00    	m.nop
   1ca4a: ef 47 7f 1f  	l.ldi16.u r8, 0xffff
   1ca4e: 01 00        	c.nop
   1ca50: 00 00 00 00  	v.nop
   1ca54: 00 00 00 00 00 00    	m.nop
   1ca5a: 6f 40 80 00  	l.ldi_hi r8, 0x1
   1ca5e: 01 00        	c.nop
   1ca60: 00 00 00 00  	v.nop
   1ca64: 00 00 00 00 00 00    	m.nop
   1ca6a: 65 29 88 01  	l.andix r5, r5, 0x8
   1ca6e: 01 00        	c.nop
   1ca70: 00 00 00 00  	v.nop
   1ca74: 00 00 00 00 00 00    	m.nop
   1ca7a: 31 12 4c 02  	l.ld.w r8, [sp, 0x2c]
   1ca7e: 01 00        	c.nop
   1ca80: 00 00 00 00  	v.nop
   1ca84: 00 00 00 00 00 00    	m.nop
   1ca8a: 65 29 06 02  	l.orix r5, r5, 0x6
   1ca8e: 01 00        	c.nop
   1ca90: 00 00 00 00  	v.nop
   1ca94: 00 00 00 00 00 00    	m.nop
   1ca9a: e5 39 88 01  	l.andix r7, r7, 0x8
   1ca9e: 01 00        	c.nop
   1caa0: 00 00 00 00  	v.nop
   1caa4: 00 00 00 00 00 00    	m.nop
   1caaa: 65 29 07 02  	l.orix r5, r5, 0x7
   1caae: 01 00        	c.nop
   1cab0: 00 00 00 00  	v.nop
   1cab4: 00 00 00 00 00 00    	m.nop
   1caba: 65 29 1d 02  	l.orix r5, r5, 0x1d
   1cabe: 01 00        	c.nop
   1cac0: 00 00 00 00  	v.nop
   1cac4: 00 00 00 00 00 00    	m.nop
   1caca: 31 20 c5 00  	l.st.w r5, [r4]
   1cace: 01 00        	c.nop
   1cad0: 00 00 00 00  	v.nop
   1cad4: 00 00 00 00 00 00    	m.nop
   1cada: 31 11 48 0a  	l.ld.w r4, [sp, 0xa8]
   1cade: 01 00        	c.nop
   1cae0: 00 00 00 00  	v.nop
   1cae4: 00 00 00 00 00 00    	m.nop
   1caea: 2f 2d 01 21  	l.ldi16.s r5, 0x34
   1caee: 01 00        	c.nop
   1caf0: 00 00 00 00  	v.nop
   1caf4: 00 00 00 00 00 00    	m.nop
   1cafa: 2f 48 08 21  	l.ldi16.s r9, 0x100
   1cafe: 01 00        	c.nop
   1cb00: 00 00 00 00  	v.nop
   1cb04: 00 00 00 00 00 00    	m.nop
   1cb0a: 6f 50 80 20  	l.ldi_hi.m1 r10, 0x1
   1cb0e: 01 00        	c.nop
   1cb10: 00 00 00 00  	v.nop
   1cb14: 00 00 00 00 00 00    	m.nop
   1cb1a: a3 21 05 10  	<unknown>
   1cb1e: 85 42        	<unknown>
   1cb20: 00 00 00 00  	v.nop
   1cb24: 00 00 00 00 00 00    	m.nop
   1cb2a: 33 01 80 00  	l.movs r4, sip_id
   1cb2e: 01 00        	c.nop
   1cb30: 00 00 00 00  	v.nop
   1cb34: 00 00 00 00 00 00    	m.nop
   1cb3a: 31 30 c5 00  	l.st.w r5, [r6]
   1cb3e: 16 82        	<unknown>
   1cb40: 00 00 00 00  	v.nop
   1cb44: 00 00 00 00 00 00    	m.nop
   1cb4a: f1 12 4c 12  	l.ld.w r11, [sp, 0x12c]
   1cb4e: 01 00        	c.nop
   1cb50: 00 00 00 00  	v.nop
   1cb54: 00 00 00 00 00 00    	m.nop
   1cb5a: 2f 32 44 21  	l.ldi16.s r6, 0x888
   1cb5e: 01 00        	c.nop
   1cb60: 00 00 00 00  	v.nop
   1cb64: 00 00 00 00 00 00    	m.nop
   1cb6a: a3 11 06 18  	<unknown>
   1cb6e: 01 00        	c.nop
   1cb70: 00 00 00 00  	v.nop
   1cb74: 00 00 00 00 00 00    	m.nop
   1cb7a: f1 31 00 00  	l.ld.b r7, [r6]
   1cb7e: 01 43        	<unknown>
   1cb80: 00 00 00 00  	v.nop
   1cb84: 00 00 00 00 00 00    	m.nop
   1cb8a: 31 12 54 0c  	l.ld.w r8, [sp, 0xd4]
   1cb8e: 01 00        	c.nop
   1cb90: 00 00 00 00  	v.nop
   1cb94: 00 00 00 00 00 00    	m.nop
   1cb9a: 31 14 cb 2c  	l.st.w r11, [sp, 0x2d0]
   1cb9e: 01 00        	c.nop
   1cba0: 00 00 00 00  	v.nop
   1cba4: 00 00 00 00 00 00    	m.nop
   1cbaa: f1 12 50 0c  	l.ld.w r11, [sp, 0xd0]
   1cbae: 01 00        	c.nop
   1cbb0: 00 00 00 00  	v.nop
   1cbb4: 00 00 00 00 00 00    	m.nop
   1cbba: 71 16 50 06  	l.ld.w r25, [sp, 0x70]
   1cbbe: 01 00        	c.nop
   1cbc0: 00 00 00 00  	v.nop
   1cbc4: 00 00 00 00 00 00    	m.nop
   1cbca: e5 39 9c 01  	l.andix r7, r7, 0x1c
   1cbce: 01 00        	c.nop
   1cbd0: 00 00 00 00  	v.nop
   1cbd4: 00 00 00 00 00 00    	m.nop
   1cbda: 31 12 c8 30  	l.st.w r8, [sp, 0x308]
   1cbde: 11 44        	<unknown>
   1cbe0: 00 00 00 00  	v.nop
   1cbe4: 00 00 00 00 00 00    	m.nop
   1cbea: 31 13 c0 30  	l.st.w r0, [sp, 0x30c]
   1cbee: 01 00        	c.nop
   1cbf0: 00 00 00 00  	v.nop
   1cbf4: 00 00 00 00 00 00    	m.nop
   1cbfa: 31 16 cb 28  	l.st.w r11, [sp, 0x298]
   1cbfe: 01 00        	c.nop
   1cc00: 00 00 00 00  	v.nop
   1cc04: 00 00 00 00 00 00    	m.nop
   1cc0a: f1 12 48 12  	l.ld.w r11, [sp, 0x128]
   1cc0e: 01 00        	c.nop
   1cc10: 00 00 00 00  	v.nop
   1cc14: 00 00 00 00 00 00    	m.nop
   1cc1a: 31 15 d6 32  	l.st.w r22, [sp, 0x334]
   1cc1e: 01 00        	c.nop
   1cc20: 00 00 00 00  	v.nop
   1cc24: 00 00 00 00 00 00    	m.nop
   1cc2a: 31 14 c8 30  	l.st.w r8, [sp, 0x310]
   1cc2e: 01 00        	c.nop
   1cc30: 00 00 00 00  	v.nop
   1cc34: 00 00 00 00 00 00    	m.nop
   1cc3a: 31 14 d6 32  	l.st.w r22, [sp, 0x330]
   1cc3e: 01 00        	c.nop
   1cc40: 00 00 00 00  	v.nop
   1cc44: 00 00 00 00 00 00    	m.nop
   1cc4a: 31 15 c9 30  	l.st.w r9, [sp, 0x314]
   1cc4e: 01 00        	c.nop
   1cc50: 00 00 00 00  	v.nop
   1cc54: 00 00 00 00 00 00    	m.nop
   1cc5a: 31 16 c5 30  	l.st.w r5, [sp, 0x318]
   1cc5e: 01 00        	c.nop
   1cc60: 00 00 00 00  	v.nop
   1cc64: 00 00 00 00 00 00    	m.nop
   1cc6a: 31 17 c5 30  	l.st.w r5, [sp, 0x31c]
   1cc6e: 01 00        	c.nop
   1cc70: 00 00 00 00  	v.nop
   1cc74: 00 00 00 00 00 00    	m.nop
   1cc7a: 31 10 c9 32  	l.st.w r9, [sp, 0x320]
   1cc7e: 01 00        	c.nop
   1cc80: 00 00 00 00  	v.nop
   1cc84: 00 00 00 00 00 00    	m.nop
   1cc8a: 31 11 c5 32  	l.st.w r5, [sp, 0x324]
   1cc8e: 01 00        	c.nop
   1cc90: 00 00 00 00  	v.nop
   1cc94: 00 00 00 00 00 00    	m.nop
   1cc9a: 31 13 c0 32  	l.st.w r0, [sp, 0x32c]
   1cc9e: 01 00        	c.nop
   1cca0: 00 00 00 00  	v.nop
   1cca4: 00 00 00 00 00 00    	m.nop
   1ccaa: 31 12 ca 32  	l.st.w r10, [sp, 0x328]
   1ccae: 01 00        	c.nop
   1ccb0: 00 00 00 00  	v.nop
   1ccb4: 00 00 00 00 00 00    	m.nop
   1ccba: 31 16 c0 32  	l.st.w r0, [sp, 0x338]
   1ccbe: 01 00        	c.nop
   1ccc0: 00 00 00 00  	v.nop
   1ccc4: 00 00 00 00 00 00    	m.nop
   1ccca: 31 15 c0 2c  	l.st.w r0, [sp, 0x2d4]
   1ccce: 01 00        	c.nop
   1ccd0: 00 00 00 00  	v.nop
   1ccd4: 00 00 00 00 00 00    	m.nop
   1ccda: 31 17 d6 2e  	l.st.w r22, [sp, 0x2fc]
   1ccde: 01 00        	c.nop
   1cce0: 00 00 00 00  	v.nop
   1cce4: 00 00 00 00 00 00    	m.nop
   1ccea: 31 16 c8 2c  	l.st.w r8, [sp, 0x2d8]
   1ccee: 01 00        	c.nop
   1ccf0: 00 00 00 00  	v.nop
   1ccf4: 00 00 00 00 00 00    	m.nop
   1ccfa: 31 16 d6 2e  	l.st.w r22, [sp, 0x2f8]
   1ccfe: 01 00        	c.nop
   1cd00: 00 00 00 00  	v.nop
   1cd04: 00 00 00 00 00 00    	m.nop
   1cd0a: 31 17 c9 2c  	l.st.w r9, [sp, 0x2dc]
   1cd0e: 01 00        	c.nop
   1cd10: 00 00 00 00  	v.nop
   1cd14: 00 00 00 00 00 00    	m.nop
   1cd1a: 31 10 c5 2e  	l.st.w r5, [sp, 0x2e0]
   1cd1e: 01 00        	c.nop
   1cd20: 00 00 00 00  	v.nop
   1cd24: 00 00 00 00 00 00    	m.nop
   1cd2a: 31 11 c5 2e  	l.st.w r5, [sp, 0x2e4]
   1cd2e: 01 00        	c.nop
   1cd30: 00 00 00 00  	v.nop
   1cd34: 00 00 00 00 00 00    	m.nop
   1cd3a: 31 12 c9 2e  	l.st.w r9, [sp, 0x2e8]
   1cd3e: 01 00        	c.nop
   1cd40: 00 00 00 00  	v.nop
   1cd44: 00 00 00 00 00 00    	m.nop
   1cd4a: 31 13 c5 2e  	l.st.w r5, [sp, 0x2ec]
   1cd4e: 01 00        	c.nop
   1cd50: 00 00 00 00  	v.nop
   1cd54: 00 00 00 00 00 00    	m.nop
   1cd5a: 31 15 c0 2e  	l.st.w r0, [sp, 0x2f4]
   1cd5e: 01 00        	c.nop
   1cd60: 00 00 00 00  	v.nop
   1cd64: 00 00 00 00 00 00    	m.nop
   1cd6a: 31 14 ca 2e  	l.st.w r10, [sp, 0x2f0]
   1cd6e: 01 00        	c.nop
   1cd70: 00 00 00 00  	v.nop
   1cd74: 00 00 00 00 00 00    	m.nop
   1cd7a: 31 10 c0 30  	l.st.w r0, [sp, 0x300]
   1cd7e: 01 00        	c.nop
   1cd80: 00 00 00 00  	v.nop
   1cd84: 00 00 00 00 00 00    	m.nop
   1cd8a: 31 17 c0 28  	l.st.w r0, [sp, 0x29c]
   1cd8e: 01 00        	c.nop
   1cd90: 00 00 00 00  	v.nop
   1cd94: 00 00 00 00 00 00    	m.nop
   1cd9a: 31 11 d6 2c  	l.st.w r22, [sp, 0x2c4]
   1cd9e: 01 00        	c.nop
   1cda0: 00 00 00 00  	v.nop
   1cda4: 00 00 00 00 00 00    	m.nop
   1cdaa: 31 10 c8 2a  	l.st.w r8, [sp, 0x2a0]
   1cdae: 01 00        	c.nop
   1cdb0: 00 00 00 00  	v.nop
   1cdb4: 00 00 00 00 00 00    	m.nop
   1cdba: 31 10 d6 2c  	l.st.w r22, [sp, 0x2c0]
   1cdbe: 01 00        	c.nop
   1cdc0: 00 00 00 00  	v.nop
   1cdc4: 00 00 00 00 00 00    	m.nop
   1cdca: 31 11 c9 2a  	l.st.w r9, [sp, 0x2a4]
   1cdce: 01 00        	c.nop
   1cdd0: 00 00 00 00  	v.nop
   1cdd4: 00 00 00 00 00 00    	m.nop
   1cdda: 31 12 c5 2a  	l.st.w r5, [sp, 0x2a8]
   1cdde: 01 00        	c.nop
   1cde0: 00 00 00 00  	v.nop
   1cde4: 00 00 00 00 00 00    	m.nop
   1cdea: 31 13 c5 2a  	l.st.w r5, [sp, 0x2ac]
   1cdee: 01 00        	c.nop
   1cdf0: 00 00 00 00  	v.nop
   1cdf4: 00 00 00 00 00 00    	m.nop
   1cdfa: 31 14 c9 2a  	l.st.w r9, [sp, 0x2b0]
   1cdfe: 01 00        	c.nop
   1ce00: 00 00 00 00  	v.nop
   1ce04: 00 00 00 00 00 00    	m.nop
   1ce0a: 31 15 c5 2a  	l.st.w r5, [sp, 0x2b4]
   1ce0e: 01 00        	c.nop
   1ce10: 00 00 00 00  	v.nop
   1ce14: 00 00 00 00 00 00    	m.nop
   1ce1a: 31 17 c0 2a  	l.st.w r0, [sp, 0x2bc]
   1ce1e: 01 00        	c.nop
   1ce20: 00 00 00 00  	v.nop
   1ce24: 00 00 00 00 00 00    	m.nop
   1ce2a: 31 16 ca 2a  	l.st.w r10, [sp, 0x2b8]
   1ce2e: 01 00        	c.nop
   1ce30: 00 00 00 00  	v.nop
   1ce34: 00 00 00 00 00 00    	m.nop
   1ce3a: 31 12 c0 2c  	l.st.w r0, [sp, 0x2c8]
   1ce3e: 01 00        	c.nop
   1ce40: 00 00 00 00  	v.nop
   1ce44: 00 00 00 00 00 00    	m.nop
   1ce4a: 31 11 c0 26  	l.st.w r0, [sp, 0x264]
   1ce4e: 01 00        	c.nop
   1ce50: 00 00 00 00  	v.nop
   1ce54: 00 00 00 00 00 00    	m.nop
   1ce5a: 31 10 cb 26  	l.st.w r11, [sp, 0x260]
   1ce5e: 01 00        	c.nop
   1ce60: 00 00 00 00  	v.nop
   1ce64: 00 00 00 00 00 00    	m.nop
   1ce6a: 31 13 d6 28  	l.st.w r22, [sp, 0x28c]
   1ce6e: 01 00        	c.nop
   1ce70: 00 00 00 00  	v.nop
   1ce74: 00 00 00 00 00 00    	m.nop
   1ce7a: 31 12 c8 26  	l.st.w r8, [sp, 0x268]
   1ce7e: 01 00        	c.nop
   1ce80: 00 00 00 00  	v.nop
   1ce84: 00 00 00 00 00 00    	m.nop
   1ce8a: 31 12 d6 28  	l.st.w r22, [sp, 0x288]
   1ce8e: 01 00        	c.nop
   1ce90: 00 00 00 00  	v.nop
   1ce94: 00 00 00 00 00 00    	m.nop
   1ce9a: 31 13 c9 26  	l.st.w r9, [sp, 0x26c]
   1ce9e: 01 00        	c.nop
   1cea0: 00 00 00 00  	v.nop
   1cea4: 00 00 00 00 00 00    	m.nop
   1ceaa: 31 14 c5 26  	l.st.w r5, [sp, 0x270]
   1ceae: 01 00        	c.nop
   1ceb0: 00 00 00 00  	v.nop
   1ceb4: 00 00 00 00 00 00    	m.nop
   1ceba: 31 15 c5 26  	l.st.w r5, [sp, 0x274]
   1cebe: 01 00        	c.nop
   1cec0: 00 00 00 00  	v.nop
   1cec4: 00 00 00 00 00 00    	m.nop
   1ceca: 31 16 c9 26  	l.st.w r9, [sp, 0x278]
   1cece: 01 00        	c.nop
   1ced0: 00 00 00 00  	v.nop
   1ced4: 00 00 00 00 00 00    	m.nop
   1ceda: 31 17 c5 26  	l.st.w r5, [sp, 0x27c]
   1cede: 01 00        	c.nop
   1cee0: 00 00 00 00  	v.nop
   1cee4: 00 00 00 00 00 00    	m.nop
   1ceea: 31 11 c0 28  	l.st.w r0, [sp, 0x284]
   1ceee: 01 00        	c.nop
   1cef0: 00 00 00 00  	v.nop
   1cef4: 00 00 00 00 00 00    	m.nop
   1cefa: 31 10 ca 28  	l.st.w r10, [sp, 0x280]
   1cefe: 01 00        	c.nop
   1cf00: 00 00 00 00  	v.nop
   1cf04: 00 00 00 00 00 00    	m.nop
   1cf0a: 31 14 c0 28  	l.st.w r0, [sp, 0x290]
   1cf0e: 01 00        	c.nop
   1cf10: 00 00 00 00  	v.nop
   1cf14: 00 00 00 00 00 00    	m.nop
   1cf1a: bb 38 00 40  	<unknown>
   1cf1e: 01 00        	c.nop
   1cf20: 00 00 00 00  	v.nop
   1cf24: 00 00 00 00 00 00    	m.nop
   1cf2a: 00 00        	s.nop16
   1cf2c: 00 00        	s.nop16
   1cf2e: 1a 82        	<unknown>
   1cf30: 00 00 00 00  	v.nop
   1cf34: 00 00 00 00 00 00    	m.nop
   1cf3a: 2f 47 4a 21  	l.ldi16.s r8, 0x95c
   1cf3e: d1 43        	<unknown>
   1cf40: 00 00 00 00  	v.nop
   1cf44: 00 00 00 00 00 00    	m.nop
   1cf4a: 23 12 08 18  	<unknown>
   1cf4e: 01 00        	c.nop
   1cf50: 00 00 00 00  	v.nop
   1cf54: 00 00 00 00 00 00    	m.nop
   1cf5a: 31 40 c7 00  	l.st.w r7, [r8]
   1cf5e: 96 83        	<unknown>
   1cf60: 00 00 00 00  	v.nop
   1cf64: 00 00 00 00 00 00    	m.nop
   1cf6a: 31 12 54 04  	l.ld.w r8, [sp, 0x54]
   1cf6e: 01 00        	c.nop
   1cf70: 00 00 00 00  	v.nop
   1cf74: 00 00 00 00 00 00    	m.nop
   1cf7a: 2f 4f 4b 21  	l.ldi16.s r9, 0x97c
   1cf7e: 01 00        	c.nop
   1cf80: 00 00 00 00  	v.nop
   1cf84: 00 00 00 00 00 00    	m.nop
   1cf8a: 63 12 09 18  	<unknown>
   1cf8e: 01 00        	c.nop
   1cf90: 00 00 00 00  	v.nop
   1cf94: 00 00 00 00 00 00    	m.nop
   1cf9a: 31 13 54 0c  	l.ld.w r12, [sp, 0xd4]
   1cf9e: 01 00        	c.nop
   1cfa0: 00 00 00 00  	v.nop
   1cfa4: 00 00 00 00 00 00    	m.nop
   1cfaa: 2f 50 4d 21  	l.ldi16.s r10, 0x9a0
   1cfae: 01 00        	c.nop
   1cfb0: 00 00 00 00  	v.nop
   1cfb4: 00 00 00 00 00 00    	m.nop
   1cfba: 31 48 c8 00  	l.st.w r8, [r9]
   1cfbe: 01 00        	c.nop
   1cfc0: 00 00 00 00  	v.nop
   1cfc4: 00 00 00 00 00 00    	m.nop
   1cfca: 71 12 5c 04  	l.ld.w r9, [sp, 0x5c]
   1cfce: 01 00        	c.nop
   1cfd0: 00 00 00 00  	v.nop
   1cfd4: 00 00 00 00 00 00    	m.nop
   1cfda: 2f 46 4b 21  	l.ldi16.s r8, 0x978
   1cfde: 01 00        	c.nop
   1cfe0: 00 00 00 00  	v.nop
   1cfe4: 00 00 00 00 00 00    	m.nop
   1cfea: 23 12 08 18  	<unknown>
   1cfee: 01 00        	c.nop
   1cff0: 00 00 00 00  	v.nop
   1cff4: 00 00 00 00 00 00    	m.nop
   1cffa: 2f 6c 4d 21  	l.ldi16.s r13, 0x9b0
   1cffe: 01 00        	c.nop
   1d000: 00 00 00 00  	v.nop
   1d004: 00 00 00 00 00 00    	m.nop
   1d00a: 31 40 c9 00  	l.st.w r9, [r8]
   1d00e: 01 00        	c.nop
   1d010: 00 00 00 00  	v.nop
   1d014: 00 00 00 00 00 00    	m.nop
   1d01a: 2f 45 4b 21  	l.ldi16.s r8, 0x974
   1d01e: 01 00        	c.nop
   1d020: 00 00 00 00  	v.nop
   1d024: 00 00 00 00 00 00    	m.nop
   1d02a: 23 12 08 18  	<unknown>
   1d02e: 01 00        	c.nop
   1d030: 00 00 00 00  	v.nop
   1d034: 00 00 00 00 00 00    	m.nop
   1d03a: 31 40 c6 00  	l.st.w r6, [r8]
   1d03e: 01 00        	c.nop
   1d040: 00 00 00 00  	v.nop
   1d044: 00 00 00 00 00 00    	m.nop
   1d04a: 2f 44 4b 21  	l.ldi16.s r8, 0x970
   1d04e: 01 00        	c.nop
   1d050: 00 00 00 00  	v.nop
   1d054: 00 00 00 00 00 00    	m.nop
   1d05a: 23 12 08 18  	<unknown>
   1d05e: 01 00        	c.nop
   1d060: 00 00 00 00  	v.nop
   1d064: 00 00 00 00 00 00    	m.nop
   1d06a: 31 40 cc 00  	l.st.w r12, [r8]
   1d06e: 01 00        	c.nop
   1d070: 00 00 00 00  	v.nop
   1d074: 00 00 00 00 00 00    	m.nop
   1d07a: 2f 41 4b 21  	l.ldi16.s r8, 0x964
   1d07e: 01 00        	c.nop
   1d080: 00 00 00 00  	v.nop
   1d084: 00 00 00 00 00 00    	m.nop
   1d08a: 23 12 08 18  	<unknown>
   1d08e: 01 00        	c.nop
   1d090: 00 00 00 00  	v.nop
   1d094: 00 00 00 00 00 00    	m.nop
   1d09a: 31 40 c4 00  	l.st.w r4, [r8]
   1d09e: 01 00        	c.nop
   1d0a0: 00 00 00 00  	v.nop
   1d0a4: 00 00 00 00 00 00    	m.nop
   1d0aa: 2f 26 48 21  	l.ldi16.s r4, 0x918
   1d0ae: 01 00        	c.nop
   1d0b0: 00 00 00 00  	v.nop
   1d0b4: 00 00 00 00 00 00    	m.nop
   1d0ba: 23 11 04 18  	<unknown>
   1d0be: 01 00        	c.nop
   1d0c0: 00 00 00 00  	v.nop
   1d0c4: 00 00 00 00 00 00    	m.nop
   1d0ca: 2f 47 52 21  	l.ldi16.s r8, 0xa5c
   1d0ce: 01 00        	c.nop
   1d0d0: 00 00 00 00  	v.nop
   1d0d4: 00 00 00 00 00 00    	m.nop
   1d0da: 23 12 08 18  	<unknown>
   1d0de: 01 00        	c.nop
   1d0e0: 00 00 00 00  	v.nop
   1d0e4: 00 00 00 00 00 00    	m.nop
   1d0ea: 31 40 c6 00  	l.st.w r6, [r8]
   1d0ee: 01 00        	c.nop
   1d0f0: 00 00 00 00  	v.nop
   1d0f4: 00 00 00 00 00 00    	m.nop
   1d0fa: 2f 40 4b 21  	l.ldi16.s r8, 0x960
   1d0fe: 01 00        	c.nop
   1d100: 00 00 00 00  	v.nop
   1d104: 00 00 00 00 00 00    	m.nop
   1d10a: 23 12 08 18  	<unknown>
   1d10e: 01 00        	c.nop
   1d110: 00 00 00 00  	v.nop
   1d114: 00 00 00 00 00 00    	m.nop
   1d11a: 31 40 d6 00  	l.st.w r22, [r8]
   1d11e: 11 44        	<unknown>
   1d120: 00 00 00 00  	v.nop
   1d124: 00 00 00 00 00 00    	m.nop
   1d12a: a3 12 0a 18  	<unknown>
   1d12e: 01 00        	c.nop
   1d130: 00 00 00 00  	v.nop
   1d134: 00 00 00 00 00 00    	m.nop
   1d13a: 31 50 c8 00  	l.st.w r8, [r10]
   1d13e: 01 00        	c.nop
   1d140: 00 00 00 00  	v.nop
   1d144: 00 00 00 00 00 00    	m.nop
   1d14a: 2f 41 4d 21  	l.ldi16.s r8, 0x9a4
   1d14e: 01 00        	c.nop
   1d150: 00 00 00 00  	v.nop
   1d154: 00 00 00 00 00 00    	m.nop
   1d15a: 23 12 08 18  	<unknown>
   1d15e: 01 00        	c.nop
   1d160: 00 00 00 00  	v.nop
   1d164: 00 00 00 00 00 00    	m.nop
   1d16a: 31 40 c5 00  	l.st.w r5, [r8]
   1d16e: 01 00        	c.nop
   1d170: 00 00 00 00  	v.nop
   1d174: 00 00 00 00 00 00    	m.nop
   1d17a: 2f 46 4d 21  	l.ldi16.s r8, 0x9b8
   1d17e: 01 00        	c.nop
   1d180: 00 00 00 00  	v.nop
   1d184: 00 00 00 00 00 00    	m.nop
   1d18a: 23 12 08 18  	<unknown>
   1d18e: 01 00        	c.nop
   1d190: 00 00 00 00  	v.nop
   1d194: 00 00 00 00 00 00    	m.nop
   1d19a: 31 40 c5 00  	l.st.w r5, [r8]
   1d19e: 01 00        	c.nop
   1d1a0: 00 00 00 00  	v.nop
   1d1a4: 00 00 00 00 00 00    	m.nop
   1d1aa: 2f 43 4e 21  	l.ldi16.s r8, 0x9cc
   1d1ae: 01 00        	c.nop
   1d1b0: 00 00 00 00  	v.nop
   1d1b4: 00 00 00 00 00 00    	m.nop
   1d1ba: 23 12 08 18  	<unknown>
   1d1be: 01 00        	c.nop
   1d1c0: 00 00 00 00  	v.nop
   1d1c4: 00 00 00 00 00 00    	m.nop
   1d1ca: 31 40 c6 00  	l.st.w r6, [r8]
   1d1ce: 01 00        	c.nop
   1d1d0: 00 00 00 00  	v.nop
   1d1d4: 00 00 00 00 00 00    	m.nop
   1d1da: 2f 45 4f 21  	l.ldi16.s r8, 0x9f4
   1d1de: 01 00        	c.nop
   1d1e0: 00 00 00 00  	v.nop
   1d1e4: 00 00 00 00 00 00    	m.nop
   1d1ea: 23 12 08 18  	<unknown>
   1d1ee: 01 00        	c.nop
   1d1f0: 00 00 00 00  	v.nop
   1d1f4: 00 00 00 00 00 00    	m.nop
   1d1fa: 31 40 c6 00  	l.st.w r6, [r8]
   1d1fe: 01 00        	c.nop
   1d200: 00 00 00 00  	v.nop
   1d204: 00 00 00 00 00 00    	m.nop
   1d20a: 31 12 40 02  	l.ld.w r8, [sp, 0x20]
   1d20e: 01 00        	c.nop
   1d210: 00 00 00 00  	v.nop
   1d214: 00 00 00 00 00 00    	m.nop
   1d21a: 2f 52 4d 21  	l.ldi16.s r10, 0x9a8
   1d21e: 01 00        	c.nop
   1d220: 00 00 00 00  	v.nop
   1d224: 00 00 00 00 00 00    	m.nop
   1d22a: a3 12 0a 18  	<unknown>
   1d22e: 01 00        	c.nop
   1d230: 00 00 00 00  	v.nop
   1d234: 00 00 00 00 00 00    	m.nop
   1d23a: 31 21 40 00  	l.ld.w r4, [r4]
   1d23e: 01 00        	c.nop
   1d240: 00 00 00 00  	v.nop
   1d244: 00 00 00 00 00 00    	m.nop
   1d24a: 31 50 c8 00  	l.st.w r8, [r10]
   1d24e: 01 00        	c.nop
   1d250: 00 00 00 00  	v.nop
   1d254: 00 00 00 00 00 00    	m.nop
   1d25a: 2f 40 08 21  	l.ldi16.s r8, 0x100
   1d25e: 01 00        	c.nop
   1d260: 00 00 00 00  	v.nop
   1d264: 00 00 00 00 00 00    	m.nop
   1d26a: 2f 47 4d 21  	l.ldi16.s r8, 0x9bc
   1d26e: a2 85        	<unknown>
   1d270: 00 00 00 00  	v.nop
   1d274: 00 00 00 00 00 00    	m.nop
   1d27a: 23 12 08 18  	<unknown>
   1d27e: 01 00        	c.nop
   1d280: 00 00 00 00  	v.nop
   1d284: 00 00 00 00 00 00    	m.nop
   1d28a: 31 40 cb 00  	l.st.w r11, [r8]
   1d28e: 01 00        	c.nop
   1d290: 00 00 00 00  	v.nop
   1d294: 00 00 00 00 00 00    	m.nop
   1d29a: 31 12 44 08  	l.ld.w r8, [sp, 0x84]
   1d29e: 01 00        	c.nop
   1d2a0: 00 00 00 00  	v.nop
   1d2a4: 00 00 00 00 00 00    	m.nop
   1d2aa: 2f 54 4e 21  	l.ldi16.s r10, 0x9d0
   1d2ae: 01 00        	c.nop
   1d2b0: 00 00 00 00  	v.nop
   1d2b4: 00 00 00 00 00 00    	m.nop
   1d2ba: a3 12 0a 18  	<unknown>
   1d2be: 01 00        	c.nop
   1d2c0: 00 00 00 00  	v.nop
   1d2c4: 00 00 00 00 00 00    	m.nop
   1d2ca: 31 50 c8 00  	l.st.w r8, [r10]
   1d2ce: 01 00        	c.nop
   1d2d0: 00 00 00 00  	v.nop
   1d2d4: 00 00 00 00 00 00    	m.nop
   1d2da: 31 12 58 10  	l.ld.w r8, [sp, 0x118]
   1d2de: 01 00        	c.nop
   1d2e0: 00 00 00 00  	v.nop
   1d2e4: 00 00 00 00 00 00    	m.nop
   1d2ea: 2f 56 4f 21  	l.ldi16.s r10, 0x9f8
   1d2ee: 01 00        	c.nop
   1d2f0: 00 00 00 00  	v.nop
   1d2f4: 00 00 00 00 00 00    	m.nop
   1d2fa: a3 12 0a 18  	<unknown>
   1d2fe: 01 00        	c.nop
   1d300: 00 00 00 00  	v.nop
   1d304: 00 00 00 00 00 00    	m.nop
   1d30a: 31 50 c8 00  	l.st.w r8, [r10]
   1d30e: 01 00        	c.nop
   1d310: 00 00 00 00  	v.nop
   1d314: 00 00 00 00 00 00    	m.nop
   1d31a: b1 12 48 02  	l.ld.w r10, [sp, 0x28]
   1d31e: 01 00        	c.nop
   1d320: 00 00 00 00  	v.nop
   1d324: 00 00 00 00 00 00    	m.nop
   1d32a: 2f 43 4d 21  	l.ldi16.s r8, 0x9ac
   1d32e: 01 00        	c.nop
   1d330: 00 00 00 00  	v.nop
   1d334: 00 00 00 00 00 00    	m.nop
   1d33a: 23 12 08 18  	<unknown>
   1d33e: 01 00        	c.nop
   1d340: 00 00 00 00  	v.nop
   1d344: 00 00 00 00 00 00    	m.nop
   1d34a: 31 40 ca 00  	l.st.w r10, [r8]
   1d34e: 01 00        	c.nop
   1d350: 00 00 00 00  	v.nop
   1d354: 00 00 00 00 00 00    	m.nop
   1d35a: 27 12 a7 09  	l.addi_sft.s.sft0 r8, sp, 0x27
   1d35e: 01 00        	c.nop
   1d360: 00 00 00 00  	v.nop
   1d364: 00 00 00 00 00 00    	m.nop
   1d36a: 31 40 c5 00  	l.st.w r5, [r8]
   1d36e: 01 00        	c.nop
   1d370: 00 00 00 00  	v.nop
   1d374: 00 00 00 00 00 00    	m.nop
   1d37a: 2f 45 4e 21  	l.ldi16.s r8, 0x9d4
   1d37e: 01 00        	c.nop
   1d380: 00 00 00 00  	v.nop
   1d384: 00 00 00 00 00 00    	m.nop
   1d38a: 23 12 08 18  	<unknown>
   1d38e: 01 00        	c.nop
   1d390: 00 00 00 00  	v.nop
   1d394: 00 00 00 00 00 00    	m.nop
   1d39a: 31 40 c6 00  	l.st.w r6, [r8]
   1d39e: 01 00        	c.nop
   1d3a0: 00 00 00 00  	v.nop
   1d3a4: 00 00 00 00 00 00    	m.nop
   1d3aa: 2f 47 4f 21  	l.ldi16.s r8, 0x9fc
   1d3ae: 01 00        	c.nop
   1d3b0: 00 00 00 00  	v.nop
   1d3b4: 00 00 00 00 00 00    	m.nop
   1d3ba: 23 12 08 18  	<unknown>
   1d3be: 01 00        	c.nop
   1d3c0: 00 00 00 00  	v.nop
   1d3c4: 00 00 00 00 00 00    	m.nop
   1d3ca: 31 40 c6 00  	l.st.w r6, [r8]
   1d3ce: 01 00        	c.nop
   1d3d0: 00 00 00 00  	v.nop
   1d3d4: 00 00 00 00 00 00    	m.nop
   1d3da: 31 12 50 02  	l.ld.w r8, [sp, 0x30]
   1d3de: 01 00        	c.nop
   1d3e0: 00 00 00 00  	v.nop
   1d3e4: 00 00 00 00 00 00    	m.nop
   1d3ea: 63 13 0d 18  	<unknown>
   1d3ee: 01 00        	c.nop
   1d3f0: 00 00 00 00  	v.nop
   1d3f4: 00 00 00 00 00 00    	m.nop
   1d3fa: 31 68 c8 00  	l.st.w r8, [r13]
   1d3fe: 01 00        	c.nop
   1d400: 00 00 00 00  	v.nop
   1d404: 00 00 00 00 00 00    	m.nop
   1d40a: 2f 41 4e 21  	l.ldi16.s r8, 0x9c4
   1d40e: 01 00        	c.nop
   1d410: 00 00 00 00  	v.nop
   1d414: 00 00 00 00 00 00    	m.nop
   1d41a: 23 12 08 18  	<unknown>
   1d41e: 01 00        	c.nop
   1d420: 00 00 00 00  	v.nop
   1d424: 00 00 00 00 00 00    	m.nop
   1d42a: 31 40 c5 00  	l.st.w r5, [r8]
   1d42e: 01 00        	c.nop
   1d430: 00 00 00 00  	v.nop
   1d434: 00 00 00 00 00 00    	m.nop
   1d43a: 31 12 58 02  	l.ld.w r8, [sp, 0x38]
   1d43e: 01 00        	c.nop
   1d440: 00 00 00 00  	v.nop
   1d444: 00 00 00 00 00 00    	m.nop
   1d44a: 2f 6e 4e 21  	l.ldi16.s r13, 0x9d8
   1d44e: 01 00        	c.nop
   1d450: 00 00 00 00  	v.nop
   1d454: 00 00 00 00 00 00    	m.nop
   1d45a: 63 13 0d 18  	<unknown>
   1d45e: 01 00        	c.nop
   1d460: 00 00 00 00  	v.nop
   1d464: 00 00 00 00 00 00    	m.nop
   1d46a: 31 68 c8 00  	l.st.w r8, [r13]
   1d46e: 01 00        	c.nop
   1d470: 00 00 00 00  	v.nop
   1d474: 00 00 00 00 00 00    	m.nop
   1d47a: 31 12 40 04  	l.ld.w r8, [sp, 0x40]
   1d47e: 01 00        	c.nop
   1d480: 00 00 00 00  	v.nop
   1d484: 00 00 00 00 00 00    	m.nop
   1d48a: 67 13 a8 09  	l.addi_sft.s.sft0 r13, sp, 0x28
   1d48e: 01 00        	c.nop
   1d490: 00 00 00 00  	v.nop
   1d494: 00 00 00 00 00 00    	m.nop
   1d49a: 31 68 c8 00  	l.st.w r8, [r13]
   1d49e: 01 00        	c.nop
   1d4a0: 00 00 00 00  	v.nop
   1d4a4: 00 00 00 00 00 00    	m.nop
   1d4aa: 31 12 54 02  	l.ld.w r8, [sp, 0x34]
   1d4ae: 01 00        	c.nop
   1d4b0: 00 00 00 00  	v.nop
   1d4b4: 00 00 00 00 00 00    	m.nop
   1d4ba: 2f 6d 4d 21  	l.ldi16.s r13, 0x9b4
   1d4be: 01 00        	c.nop
   1d4c0: 00 00 00 00  	v.nop
   1d4c4: 00 00 00 00 00 00    	m.nop
   1d4ca: 63 13 0d 18  	<unknown>
   1d4ce: 01 00        	c.nop
   1d4d0: 00 00 00 00  	v.nop
   1d4d4: 00 00 00 00 00 00    	m.nop
   1d4da: 31 68 c8 00  	l.st.w r8, [r13]
   1d4de: 01 00        	c.nop
   1d4e0: 00 00 00 00  	v.nop
   1d4e4: 00 00 00 00 00 00    	m.nop
   1d4ea: 2f 42 4e 21  	l.ldi16.s r8, 0x9c8
   1d4ee: 01 00        	c.nop
   1d4f0: 00 00 00 00  	v.nop
   1d4f4: 00 00 00 00 00 00    	m.nop
   1d4fa: 23 12 08 18  	<unknown>
   1d4fe: 01 00        	c.nop
   1d500: 00 00 00 00  	v.nop
   1d504: 00 00 00 00 00 00    	m.nop
   1d50a: 31 40 cb 00  	l.st.w r11, [r8]
   1d50e: 01 00        	c.nop
   1d510: 00 00 00 00  	v.nop
   1d514: 00 00 00 00 00 00    	m.nop
   1d51a: 2f 47 4e 21  	l.ldi16.s r8, 0x9dc
   1d51e: 01 00        	c.nop
   1d520: 00 00 00 00  	v.nop
   1d524: 00 00 00 00 00 00    	m.nop
   1d52a: 23 12 08 18  	<unknown>
   1d52e: 01 00        	c.nop
   1d530: 00 00 00 00  	v.nop
   1d534: 00 00 00 00 00 00    	m.nop
   1d53a: 31 40 c6 00  	l.st.w r6, [r8]
   1d53e: 01 00        	c.nop
   1d540: 00 00 00 00  	v.nop
   1d544: 00 00 00 00 00 00    	m.nop
   1d54a: 2f 41 50 21  	l.ldi16.s r8, 0xa04
   1d54e: 01 00        	c.nop
   1d550: 00 00 00 00  	v.nop
   1d554: 00 00 00 00 00 00    	m.nop
   1d55a: 23 12 08 18  	<unknown>
   1d55e: 01 00        	c.nop
   1d560: 00 00 00 00  	v.nop
   1d564: 00 00 00 00 00 00    	m.nop
   1d56a: 31 40 c6 00  	l.st.w r6, [r8]
   1d56e: 71 44        	<unknown>
   1d570: 00 00 00 00  	v.nop
   1d574: 00 00 00 00 00 00    	m.nop
   1d57a: 23 22 08 10  	<unknown>
   1d57e: 01 00        	c.nop
   1d580: 00 00 00 00  	v.nop
   1d584: 00 00 00 00 00 00    	m.nop
   1d58a: 31 40 c9 00  	l.st.w r9, [r8]
   1d58e: 01 00        	c.nop
   1d590: 00 00 00 00  	v.nop
   1d594: 00 00 00 00 00 00    	m.nop
   1d59a: 71 12 54 12  	l.ld.w r9, [sp, 0x134]
   1d59e: 01 00        	c.nop
   1d5a0: 00 00 00 00  	v.nop
   1d5a4: 00 00 00 00 00 00    	m.nop
   1d5aa: 31 47 c9 fe  	l.st.w r9, [r8, -0x4]
   1d5ae: 01 00        	c.nop
   1d5b0: 00 00 00 00  	v.nop
   1d5b4: 00 00 00 00 00 00    	m.nop
   1d5ba: 31 42 cc 00  	l.st.w r12, [r8, 0x8]
   1d5be: 01 00        	c.nop
   1d5c0: 00 00 00 00  	v.nop
   1d5c4: 00 00 00 00 00 00    	m.nop
   1d5ca: 31 41 c6 00  	l.st.w r6, [r8, 0x4]
   1d5ce: 01 00        	c.nop
   1d5d0: 00 00 00 00  	v.nop
   1d5d4: 00 00 00 00 00 00    	m.nop
   1d5da: 31 42 de 0a  	l.st.w r30, [r8, 0xa8]
   1d5de: 01 00        	c.nop
   1d5e0: 00 00 00 00  	v.nop
   1d5e4: 00 00 00 00 00 00    	m.nop
   1d5ea: 71 12 48 10  	l.ld.w r9, [sp, 0x108]
   1d5ee: 01 00        	c.nop
   1d5f0: 00 00 00 00  	v.nop
   1d5f4: 00 00 00 00 00 00    	m.nop
   1d5fa: 31 43 c9 0a  	l.st.w r9, [r8, 0xac]
   1d5fe: 01 00        	c.nop
   1d600: 00 00 00 00  	v.nop
   1d604: 00 00 00 00 00 00    	m.nop
   1d60a: 31 44 ca 0a  	l.st.w r10, [r8, 0xb0]
   1d60e: 01 00        	c.nop
   1d610: 00 00 00 00  	v.nop
   1d614: 00 00 00 00 00 00    	m.nop
   1d61a: 71 12 4c 08  	l.ld.w r9, [sp, 0x8c]
   1d61e: 01 00        	c.nop
   1d620: 00 00 00 00  	v.nop
   1d624: 00 00 00 00 00 00    	m.nop
   1d62a: 31 45 c9 0a  	l.st.w r9, [r8, 0xb4]
   1d62e: 01 00        	c.nop
   1d630: 00 00 00 00  	v.nop
   1d634: 00 00 00 00 00 00    	m.nop
   1d63a: 31 44 c5 00  	l.st.w r5, [r8, 0x10]
   1d63e: 01 00        	c.nop
   1d640: 00 00 00 00  	v.nop
   1d644: 00 00 00 00 00 00    	m.nop
   1d64a: 31 41 c6 0c  	l.st.w r6, [r8, 0xc4]
   1d64e: 01 00        	c.nop
   1d650: 00 00 00 00  	v.nop
   1d654: 00 00 00 00 00 00    	m.nop
   1d65a: 71 12 54 10  	l.ld.w r9, [sp, 0x114]
   1d65e: 01 00        	c.nop
   1d660: 00 00 00 00  	v.nop
   1d664: 00 00 00 00 00 00    	m.nop
   1d66a: 31 42 c9 0c  	l.st.w r9, [r8, 0xc8]
   1d66e: 01 00        	c.nop
   1d670: 00 00 00 00  	v.nop
   1d674: 00 00 00 00 00 00    	m.nop
   1d67a: 31 43 c6 0c  	l.st.w r6, [r8, 0xcc]
   1d67e: 01 00        	c.nop
   1d680: 00 00 00 00  	v.nop
   1d684: 00 00 00 00 00 00    	m.nop
   1d68a: 71 12 50 10  	l.ld.w r9, [sp, 0x110]
   1d68e: 01 00        	c.nop
   1d690: 00 00 00 00  	v.nop
   1d694: 00 00 00 00 00 00    	m.nop
   1d69a: 31 41 c9 06  	l.st.w r9, [r8, 0x64]
   1d69e: 01 00        	c.nop
   1d6a0: 00 00 00 00  	v.nop
   1d6a4: 00 00 00 00 00 00    	m.nop
   1d6aa: 31 41 cb 0a  	l.st.w r11, [r8, 0xa4]
   1d6ae: 01 00        	c.nop
   1d6b0: 00 00 00 00  	v.nop
   1d6b4: 00 00 00 00 00 00    	m.nop
   1d6ba: 31 46 c5 0a  	l.st.w r5, [r8, 0xb8]
   1d6be: 01 00        	c.nop
   1d6c0: 00 00 00 00  	v.nop
   1d6c4: 00 00 00 00 00 00    	m.nop
   1d6ca: 31 47 c5 0a  	l.st.w r5, [r8, 0xbc]
   1d6ce: 01 00        	c.nop
   1d6d0: 00 00 00 00  	v.nop
   1d6d4: 00 00 00 00 00 00    	m.nop
   1d6da: 31 40 cb 0c  	l.st.w r11, [r8, 0xc0]
   1d6de: 01 00        	c.nop
   1d6e0: 00 00 00 00  	v.nop
   1d6e4: 00 00 00 00 00 00    	m.nop
   1d6ea: 31 44 c6 0c  	l.st.w r6, [r8, 0xd0]
   1d6ee: 01 00        	c.nop
   1d6f0: 00 00 00 00  	v.nop
   1d6f4: 00 00 00 00 00 00    	m.nop
   1d6fa: 71 12 4c 0a  	l.ld.w r9, [sp, 0xac]
   1d6fe: 01 00        	c.nop
   1d700: 00 00 00 00  	v.nop
   1d704: 00 00 00 00 00 00    	m.nop
   1d70a: 31 46 c9 fe  	l.st.w r9, [r8, -0x8]
   1d70e: 01 00        	c.nop
   1d710: 00 00 00 00  	v.nop
   1d714: 00 00 00 00 00 00    	m.nop
   1d71a: 31 12 08 36  	l.ld.b r8, [sp, 0x368]
   1d71e: 01 00        	c.nop
   1d720: 00 00 00 00  	v.nop
   1d724: 00 00 00 00 00 00    	m.nop
   1d72a: 25 42 9c 01  	l.andix r8, r8, 0x1c
   1d72e: 01 00        	c.nop
   1d730: 00 00 00 00  	v.nop
   1d734: 00 00 00 00 00 00    	m.nop
   1d73a: bb 40 06 40  	<unknown>
   1d73e: 01 00        	c.nop
   1d740: 00 00 00 00  	v.nop
   1d744: 00 00 00 00 00 00    	m.nop
   1d74a: 00 00        	s.nop16
   1d74c: 00 00        	s.nop16
   1d74e: 9a 83        	<unknown>
   1d750: 00 00 00 00  	v.nop
   1d754: 00 00 00 00 00 00    	m.nop
   1d75a: 31 11 c7 44  	l.st.w r7, [sp, 0x444]
   1d75e: 91 44        	<unknown>
   1d760: 00 00 00 00  	v.nop
   1d764: 00 00 00 00 00 00    	m.nop
   1d76a: f1 11 58 3e  	l.ld.w r7, [sp, 0x3f8]
   1d76e: 51 44        	<unknown>
   1d770: 00 00 00 00  	v.nop
   1d774: 00 00 00 00 00 00    	m.nop
   1d77a: 31 10 c9 48  	l.st.w r9, [sp, 0x480]
   1d77e: f1 44        	<unknown>
   1d780: 00 00 00 00  	v.nop
   1d784: 00 00 00 00 00 00    	m.nop
   1d78a: 31 17 c8 42  	l.st.w r8, [sp, 0x43c]
   1d78e: 01 00        	c.nop
   1d790: 00 00 00 00  	v.nop
   1d794: 00 00 00 00 00 00    	m.nop
   1d79a: 31 12 50 04  	l.ld.w r8, [sp, 0x50]
   1d79e: 01 00        	c.nop
   1d7a0: 00 00 00 00  	v.nop
   1d7a4: 00 00 00 00 00 00    	m.nop
   1d7aa: 31 13 50 0c  	l.ld.w r12, [sp, 0xd0]
   1d7ae: 01 00        	c.nop
   1d7b0: 00 00 00 00  	v.nop
   1d7b4: 00 00 00 00 00 00    	m.nop
   1d7ba: 71 12 40 02  	l.ld.w r9, [sp, 0x20]
   1d7be: a6 93        	<unknown>
   1d7c0: 00 00 00 00  	v.nop
   1d7c4: 00 00 00 00 00 00    	m.nop
   1d7ca: b1 12 48 02  	l.ld.w r10, [sp, 0x28]
   1d7ce: 01 00        	c.nop
   1d7d0: 00 00 00 00  	v.nop
   1d7d4: 00 00 00 00 00 00    	m.nop
   1d7da: 31 15 c6 44  	l.st.w r6, [sp, 0x454]
   1d7de: 01 00        	c.nop
   1d7e0: 00 00 00 00  	v.nop
   1d7e4: 00 00 00 00 00 00    	m.nop
   1d7ea: 31 17 c8 44  	l.st.w r8, [sp, 0x45c]
   1d7ee: 01 00        	c.nop
   1d7f0: 00 00 00 00  	v.nop
   1d7f4: 00 00 00 00 00 00    	m.nop
   1d7fa: 31 12 58 04  	l.ld.w r8, [sp, 0x58]
   1d7fe: 01 00        	c.nop
   1d800: 00 00 00 00  	v.nop
   1d804: 00 00 00 00 00 00    	m.nop
   1d80a: 31 12 c9 48  	l.st.w r9, [sp, 0x488]
   1d80e: 01 00        	c.nop
   1d810: 00 00 00 00  	v.nop
   1d814: 00 00 00 00 00 00    	m.nop
   1d81a: 2f 48 08 21  	l.ldi16.s r9, 0x100
   1d81e: 01 00        	c.nop
   1d820: 00 00 00 00  	v.nop
   1d824: 00 00 00 00 00 00    	m.nop
   1d82a: 71 12 44 08  	l.ld.w r9, [sp, 0x84]
   1d82e: a6 85        	<unknown>
   1d830: 00 00 00 00  	v.nop
   1d834: 00 00 00 00 00 00    	m.nop
   1d83a: 31 14 cc 44  	l.st.w r12, [sp, 0x450]
   1d83e: 01 00        	c.nop
   1d840: 00 00 00 00  	v.nop
   1d844: 00 00 00 00 00 00    	m.nop
   1d84a: 31 16 c8 44  	l.st.w r8, [sp, 0x458]
   1d84e: 01 00        	c.nop
   1d850: 00 00 00 00  	v.nop
   1d854: 00 00 00 00 00 00    	m.nop
   1d85a: 31 17 c6 52  	l.st.w r6, [sp, 0x53c]
   1d85e: 01 00        	c.nop
   1d860: 00 00 00 00  	v.nop
   1d864: 00 00 00 00 00 00    	m.nop
   1d86a: 31 10 d6 44  	l.st.w r22, [sp, 0x440]
   1d86e: 01 00        	c.nop
   1d870: 00 00 00 00  	v.nop
   1d874: 00 00 00 00 00 00    	m.nop
   1d87a: 31 14 c9 4a  	l.st.w r9, [sp, 0x4b0]
   1d87e: 01 00        	c.nop
   1d880: 00 00 00 00  	v.nop
   1d884: 00 00 00 00 00 00    	m.nop
   1d88a: 71 12 58 10  	l.ld.w r9, [sp, 0x118]
   1d88e: 01 00        	c.nop
   1d890: 00 00 00 00  	v.nop
   1d894: 00 00 00 00 00 00    	m.nop
   1d89a: 31 11 c5 48  	l.st.w r5, [sp, 0x484]
   1d89e: 01 00        	c.nop
   1d8a0: 00 00 00 00  	v.nop
   1d8a4: 00 00 00 00 00 00    	m.nop
   1d8aa: 31 16 c5 48  	l.st.w r5, [sp, 0x498]
   1d8ae: 01 00        	c.nop
   1d8b0: 00 00 00 00  	v.nop
   1d8b4: 00 00 00 00 00 00    	m.nop
   1d8ba: 31 13 c6 4a  	l.st.w r6, [sp, 0x4ac]
   1d8be: 01 00        	c.nop
   1d8c0: 00 00 00 00  	v.nop
   1d8c4: 00 00 00 00 00 00    	m.nop
   1d8ca: 31 15 c6 4c  	l.st.w r6, [sp, 0x4d4]
   1d8ce: 01 00        	c.nop
   1d8d0: 00 00 00 00  	v.nop
   1d8d4: 00 00 00 00 00 00    	m.nop
   1d8da: 31 16 c9 4c  	l.st.w r9, [sp, 0x4d8]
   1d8de: 01 00        	c.nop
   1d8e0: 00 00 00 00  	v.nop
   1d8e4: 00 00 00 00 00 00    	m.nop
   1d8ea: 71 12 50 02  	l.ld.w r9, [sp, 0x30]
   1d8ee: 01 00        	c.nop
   1d8f0: 00 00 00 00  	v.nop
   1d8f4: 00 00 00 00 00 00    	m.nop
   1d8fa: 31 17 cb 48  	l.st.w r11, [sp, 0x49c]
   1d8fe: 01 00        	c.nop
   1d900: 00 00 00 00  	v.nop
   1d904: 00 00 00 00 00 00    	m.nop
   1d90a: 31 13 ca 48  	l.st.w r10, [sp, 0x48c]
   1d90e: 01 00        	c.nop
   1d910: 00 00 00 00  	v.nop
   1d914: 00 00 00 00 00 00    	m.nop
   1d91a: 31 10 c5 4a  	l.st.w r5, [sp, 0x4a0]
   1d91e: 01 00        	c.nop
   1d920: 00 00 00 00  	v.nop
   1d924: 00 00 00 00 00 00    	m.nop
   1d92a: 31 15 c6 4a  	l.st.w r6, [sp, 0x4b4]
   1d92e: 01 00        	c.nop
   1d930: 00 00 00 00  	v.nop
   1d934: 00 00 00 00 00 00    	m.nop
   1d93a: 31 14 c9 48  	l.st.w r9, [sp, 0x490]
   1d93e: 01 00        	c.nop
   1d940: 00 00 00 00  	v.nop
   1d944: 00 00 00 00 00 00    	m.nop
   1d94a: 71 12 58 02  	l.ld.w r9, [sp, 0x38]
   1d94e: 01 00        	c.nop
   1d950: 00 00 00 00  	v.nop
   1d954: 00 00 00 00 00 00    	m.nop
   1d95a: 31 17 c6 4c  	l.st.w r6, [sp, 0x4dc]
   1d95e: 01 00        	c.nop
   1d960: 00 00 00 00  	v.nop
   1d964: 00 00 00 00 00 00    	m.nop
   1d96a: 31 11 c5 4a  	l.st.w r5, [sp, 0x4a4]
   1d96e: 01 00        	c.nop
   1d970: 00 00 00 00  	v.nop
   1d974: 00 00 00 00 00 00    	m.nop
   1d97a: 31 12 cb 4a  	l.st.w r11, [sp, 0x4a8]
   1d97e: 01 00        	c.nop
   1d980: 00 00 00 00  	v.nop
   1d984: 00 00 00 00 00 00    	m.nop
   1d98a: 31 17 c6 4a  	l.st.w r6, [sp, 0x4bc]
   1d98e: 01 00        	c.nop
   1d990: 00 00 00 00  	v.nop
   1d994: 00 00 00 00 00 00    	m.nop
   1d99a: 31 16 c9 4a  	l.st.w r9, [sp, 0x4b8]
   1d99e: 01 00        	c.nop
   1d9a0: 00 00 00 00  	v.nop
   1d9a4: 00 00 00 00 00 00    	m.nop
   1d9aa: 71 12 40 04  	l.ld.w r9, [sp, 0x40]
   1d9ae: 01 00        	c.nop
   1d9b0: 00 00 00 00  	v.nop
   1d9b4: 00 00 00 00 00 00    	m.nop
   1d9ba: 31 11 c6 4e  	l.st.w r6, [sp, 0x4e4]
   1d9be: 01 00        	c.nop
   1d9c0: 00 00 00 00  	v.nop
   1d9c4: 00 00 00 00 00 00    	m.nop
   1d9ca: 31 10 c9 4e  	l.st.w r9, [sp, 0x4e0]
   1d9ce: 01 00        	c.nop
   1d9d0: 00 00 00 00  	v.nop
   1d9d4: 00 00 00 00 00 00    	m.nop
   1d9da: 71 12 54 02  	l.ld.w r9, [sp, 0x34]
   1d9de: 01 00        	c.nop
   1d9e0: 00 00 00 00  	v.nop
   1d9e4: 00 00 00 00 00 00    	m.nop
   1d9ea: 31 15 c9 48  	l.st.w r9, [sp, 0x494]
   1d9ee: 01 00        	c.nop
   1d9f0: 00 00 00 00  	v.nop
   1d9f4: 00 00 00 00 00 00    	m.nop
   1d9fa: 31 38 c8 00  	l.st.w r8, [r7]
   1d9fe: 01 00        	c.nop
   1da00: 00 00 00 00  	v.nop
   1da04: 00 00 00 00 00 00    	m.nop
   1da0a: 31 12 50 12  	l.ld.w r8, [sp, 0x130]
   1da0e: 01 00        	c.nop
   1da10: 00 00 00 00  	v.nop
   1da14: 00 00 00 00 00 00    	m.nop
   1da1a: 31 3f c8 fe  	l.st.w r8, [r7, -0x4]
   1da1e: 01 00        	c.nop
   1da20: 00 00 00 00  	v.nop
   1da24: 00 00 00 00 00 00    	m.nop
   1da2a: 31 3a cc 00  	l.st.w r12, [r7, 0x8]
   1da2e: 01 00        	c.nop
   1da30: 00 00 00 00  	v.nop
   1da34: 00 00 00 00 00 00    	m.nop
   1da3a: 31 39 c6 00  	l.st.w r6, [r7, 0x4]
   1da3e: 01 00        	c.nop
   1da40: 00 00 00 00  	v.nop
   1da44: 00 00 00 00 00 00    	m.nop
   1da4a: 31 3a de 0a  	l.st.w r30, [r7, 0xa8]
   1da4e: 01 00        	c.nop
   1da50: 00 00 00 00  	v.nop
   1da54: 00 00 00 00 00 00    	m.nop
   1da5a: 31 12 48 10  	l.ld.w r8, [sp, 0x108]
   1da5e: 01 00        	c.nop
   1da60: 00 00 00 00  	v.nop
   1da64: 00 00 00 00 00 00    	m.nop
   1da6a: 31 3b c8 0a  	l.st.w r8, [r7, 0xac]
   1da6e: 01 00        	c.nop
   1da70: 00 00 00 00  	v.nop
   1da74: 00 00 00 00 00 00    	m.nop
   1da7a: 31 3c ca 0a  	l.st.w r10, [r7, 0xb0]
   1da7e: 01 00        	c.nop
   1da80: 00 00 00 00  	v.nop
   1da84: 00 00 00 00 00 00    	m.nop
   1da8a: 31 12 4c 08  	l.ld.w r8, [sp, 0x8c]
   1da8e: 01 00        	c.nop
   1da90: 00 00 00 00  	v.nop
   1da94: 00 00 00 00 00 00    	m.nop
   1da9a: 31 3d c8 0a  	l.st.w r8, [r7, 0xb4]
   1da9e: 01 00        	c.nop
   1daa0: 00 00 00 00  	v.nop
   1daa4: 00 00 00 00 00 00    	m.nop
   1daaa: 31 3c c5 00  	l.st.w r5, [r7, 0x10]
   1daae: 01 00        	c.nop
   1dab0: 00 00 00 00  	v.nop
   1dab4: 00 00 00 00 00 00    	m.nop
   1daba: 31 39 c6 0c  	l.st.w r6, [r7, 0xc4]
   1dabe: 01 00        	c.nop
   1dac0: 00 00 00 00  	v.nop
   1dac4: 00 00 00 00 00 00    	m.nop
   1daca: 31 12 54 10  	l.ld.w r8, [sp, 0x114]
   1dace: 01 00        	c.nop
   1dad0: 00 00 00 00  	v.nop
   1dad4: 00 00 00 00 00 00    	m.nop
   1dada: 31 3a c8 0c  	l.st.w r8, [r7, 0xc8]
   1dade: 01 00        	c.nop
   1dae0: 00 00 00 00  	v.nop
   1dae4: 00 00 00 00 00 00    	m.nop
   1daea: 31 3b c6 0c  	l.st.w r6, [r7, 0xcc]
   1daee: 01 00        	c.nop
   1daf0: 00 00 00 00  	v.nop
   1daf4: 00 00 00 00 00 00    	m.nop
   1dafa: 31 12 50 10  	l.ld.w r8, [sp, 0x110]
   1dafe: 01 00        	c.nop
   1db00: 00 00 00 00  	v.nop
   1db04: 00 00 00 00 00 00    	m.nop
   1db0a: 31 39 c8 06  	l.st.w r8, [r7, 0x64]
   1db0e: 01 00        	c.nop
   1db10: 00 00 00 00  	v.nop
   1db14: 00 00 00 00 00 00    	m.nop
   1db1a: 31 39 cb 0a  	l.st.w r11, [r7, 0xa4]
   1db1e: 01 00        	c.nop
   1db20: 00 00 00 00  	v.nop
   1db24: 00 00 00 00 00 00    	m.nop
   1db2a: 31 3e c5 0a  	l.st.w r5, [r7, 0xb8]
   1db2e: 01 00        	c.nop
   1db30: 00 00 00 00  	v.nop
   1db34: 00 00 00 00 00 00    	m.nop
   1db3a: 31 3f c5 0a  	l.st.w r5, [r7, 0xbc]
   1db3e: 01 00        	c.nop
   1db40: 00 00 00 00  	v.nop
   1db44: 00 00 00 00 00 00    	m.nop
   1db4a: 31 38 cb 0c  	l.st.w r11, [r7, 0xc0]
   1db4e: 01 00        	c.nop
   1db50: 00 00 00 00  	v.nop
   1db54: 00 00 00 00 00 00    	m.nop
   1db5a: 31 3c c6 0c  	l.st.w r6, [r7, 0xd0]
   1db5e: 01 00        	c.nop
   1db60: 00 00 00 00  	v.nop
   1db64: 00 00 00 00 00 00    	m.nop
   1db6a: b1 11 4c 0a  	l.ld.w r6, [sp, 0xac]
   1db6e: 01 00        	c.nop
   1db70: 00 00 00 00  	v.nop
   1db74: 00 00 00 00 00 00    	m.nop
   1db7a: 31 3e c6 fe  	l.st.w r6, [r7, -0x8]
   1db7e: 01 00        	c.nop
   1db80: 00 00 00 00  	v.nop
   1db84: 00 00 00 00 00 00    	m.nop
   1db8a: 2f 34 44 21  	l.ldi16.s r6, 0x890
   1db8e: 01 00        	c.nop
   1db90: 00 00 00 00  	v.nop
   1db94: 00 00 00 00 00 00    	m.nop
   1db9a: a3 11 06 18  	<unknown>
   1db9e: 01 00        	c.nop
   1dba0: 00 00 00 00  	v.nop
   1dba4: 00 00 00 00 00 00    	m.nop
   1dbaa: b1 31 40 00  	l.ld.w r6, [r6]
   1dbae: 01 00        	c.nop
   1dbb0: 00 00 00 00  	v.nop
   1dbb4: 00 00 00 00 00 00    	m.nop
   1dbba: 7b 37 05 00  	<unknown>
   1dbbe: 01 00        	c.nop
   1dbc0: 00 00 00 00  	v.nop
   1dbc4: 00 00 00 00 00 00    	m.nop
   1dbca: fb 35 00 40  	<unknown>
   1dbce: 01 00        	c.nop
   1dbd0: 00 00 00 00  	v.nop
   1dbd4: 00 00 00 00 00 00    	m.nop
   1dbda: 73 b9 83 00  	l.movs r5, md23_wrk
   1dbde: 01 00        	c.nop
   1dbe0: 00 00 00 00  	v.nop
   1dbe4: 00 00 00 00 00 00    	m.nop
   1dbea: 65 29 96 01  	l.andix r5, r5, 0x16
   1dbee: 01 00        	c.nop
   1dbf0: 00 00 00 00  	v.nop
   1dbf4: 00 00 00 00 00 00    	m.nop
   1dbfa: 7b 2c 00 00  	<unknown>
   1dbfe: 01 00        	c.nop
   1dc00: 00 00 00 00  	v.nop
   1dc04: 00 00 00 00 00 00    	m.nop
   1dc0a: 2f 2a 04 21  	l.ldi16.s r5, 0x88
   1dc0e: 01 00        	c.nop
   1dc10: 00 00 00 00  	v.nop
   1dc14: 00 00 00 00 00 00    	m.nop
   1dc1a: 63 21 05 10  	<unknown>
   1dc1e: 01 00        	c.nop
   1dc20: 00 00 00 00  	v.nop
   1dc24: 00 00 00 00 00 00    	m.nop
   1dc2a: b1 29 40 00  	l.ld.w r6, [r5]
   1dc2e: 01 00        	c.nop
   1dc30: 00 00 00 00  	v.nop
   1dc34: 00 00 00 00 00 00    	m.nop
   1dc3a: f3 a1 83 00  	l.movs r7, md20_wrk
   1dc3e: 01 00        	c.nop
   1dc40: 00 00 00 00  	v.nop
   1dc44: 00 00 00 00 00 00    	m.nop
   1dc4a: 2f 48 f0 3e  	l.ldi_hi.m1 r9, 0xfe00
   1dc4e: 86 03        	<unknown>
   1dc50: 00 00 00 00  	v.nop
   1dc54: 00 00 00 00 00 00    	m.nop
   1dc5a: e5 39 89 01  	l.andix r7, r7, 0x9
   1dc5e: 01 00        	c.nop
   1dc60: 00 00 00 00  	v.nop
   1dc64: 00 00 00 00 00 00    	m.nop
   1dc6a: ef 4f 7f 1f  	l.ldi16.u r9, 0xffff
   1dc6e: 01 00        	c.nop
   1dc70: 00 00 00 00  	v.nop
   1dc74: 00 00 00 00 00 00    	m.nop
   1dc7a: 33 02 80 00  	l.movs r8, sip_id
   1dc7e: 01 00        	c.nop
   1dc80: 00 00 00 00  	v.nop
   1dc84: 00 00 00 00 00 00    	m.nop
   1dc8a: 6f 48 80 00  	l.ldi_hi r9, 0x1
   1dc8e: 3e 04        	<unknown>
   1dc90: 00 00 00 00  	v.nop
   1dc94: 00 00 00 00 00 00    	m.nop
   1dc9a: a5 31 89 01  	l.andix r6, r6, 0x9
   1dc9e: 01 00        	c.nop
   1dca0: 00 00 00 00  	v.nop
   1dca4: 00 00 00 00 00 00    	m.nop
   1dcaa: 71 12 4c 02  	l.ld.w r9, [sp, 0x2c]
   1dcae: 01 00        	c.nop
   1dcb0: 00 00 00 00  	v.nop
   1dcb4: 00 00 00 00 00 00    	m.nop
   1dcba: a5 31 07 02  	l.orix r6, r6, 0x7
   1dcbe: 01 00        	c.nop
   1dcc0: 00 00 00 00  	v.nop
   1dcc4: 00 00 00 00 00 00    	m.nop
   1dcca: 00 00        	s.nop16
   1dccc: 00 00        	s.nop16
   1dcce: 65 8c        	<unknown>
   1dcd0: 00 00 00 00  	v.nop
   1dcd4: 00 00 00 00 00 00    	m.nop
   1dcda: a5 31 08 02  	l.orix r6, r6, 0x8
   1dcde: 01 00        	c.nop
   1dce0: 00 00 00 00  	v.nop
   1dce4: 00 00 00 00 00 00    	m.nop
   1dcea: a5 31 1d 02  	l.orix r6, r6, 0x1d
   1dcee: 01 00        	c.nop
   1dcf0: 00 00 00 00  	v.nop
   1dcf4: 00 00 00 00 00 00    	m.nop
   1dcfa: 31 28 c6 00  	l.st.w r6, [r5]
   1dcfe: 01 00        	c.nop
   1dd00: 00 00 00 00  	v.nop
   1dd04: 00 00 00 00 00 00    	m.nop
   1dd0a: 2f 2c 44 21  	l.ldi16.s r5, 0x890
   1dd0e: 01 00        	c.nop
   1dd10: 00 00 00 00  	v.nop
   1dd14: 00 00 00 00 00 00    	m.nop
   1dd1a: 63 11 05 18  	<unknown>
   1dd1e: 01 00        	c.nop
   1dd20: 00 00 00 00  	v.nop
   1dd24: 00 00 00 00 00 00    	m.nop
   1dd2a: b1 29 40 00  	l.ld.w r6, [r5]
   1dd2e: 01 00        	c.nop
   1dd30: 00 00 00 00  	v.nop
   1dd34: 00 00 00 00 00 00    	m.nop
   1dd3a: 7b 32 1a 00  	<unknown>
   1dd3e: 01 00        	c.nop
   1dd40: 00 00 00 00  	v.nop
   1dd44: 00 00 00 00 00 00    	m.nop
   1dd4a: fb 33 00 40  	<unknown>
   1dd4e: 01 00        	c.nop
   1dd50: 00 00 00 00  	v.nop
   1dd54: 00 00 00 00 00 00    	m.nop
   1dd5a: 2f 28 08 29  	l.ldi16.s r5, 0x4100
   1dd5e: 01 00        	c.nop
   1dd60: 00 00 00 00  	v.nop
   1dd64: 00 00 00 00 00 00    	m.nop
   1dd6a: af 2f 91 00  	l.ldi_hi r5, 0x23e
   1dd6e: 01 00        	c.nop
   1dd70: 00 00 00 00  	v.nop
   1dd74: 00 00 00 00 00 00    	m.nop
   1dd7a: 2b 68 80 01  	l.jal r0, 0xd <0x1de40>
   1dd7e: 01 00        	c.nop
   1dd80: 00 00 00 00  	v.nop
   1dd84: 00 00 00 00 00 00    	m.nop
   1dd8a: 73 b9 83 00  	l.movs r5, md23_wrk
   1dd8e: 11 43        	<unknown>
   1dd90: 00 00 00 00  	v.nop
   1dd94: 00 00 00 00 00 00    	m.nop
   1dd9a: 65 29 86 01  	l.andix r5, r5, 0x6
   1dd9e: 01 00        	c.nop
   1dda0: 00 00 00 00  	v.nop
   1dda4: 00 00 00 00 00 00    	m.nop
   1ddaa: bb 29 60 7f  	<unknown>
   1ddae: 01 00        	c.nop
   1ddb0: 00 00 00 00  	v.nop
   1ddb4: 00 00 00 00 00 00    	m.nop
   1ddba: 2b 28 ff 3f  	l.jal r0, -0xb <0x1dd00>
   1ddbe: 01 00        	c.nop
   1ddc0: 00 00 00 00  	v.nop
   1ddc4: 00 00 00 00 00 00    	m.nop
   1ddca: 73 01 80 00  	l.movs r5, sip_id
   1ddce: 01 00        	c.nop
   1ddd0: 00 00 00 00  	v.nop
   1ddd4: 00 00 00 00 00 00    	m.nop
   1ddda: ef 37 80 20  	l.ldi_hi.m1 r6, 0x1f
   1ddde: c2 02        	<unknown>
   1dde0: 00 00 00 00  	v.nop
   1dde4: 00 00 00 00 00 00    	m.nop
   1ddea: 65 29 86 01  	l.andix r5, r5, 0x6
   1ddee: 01 00        	c.nop
   1ddf0: 00 00 00 00  	v.nop
   1ddf4: 00 00 00 00 00 00    	m.nop
   1ddfa: 2f 30 08 15  	l.ldi16.u r6, 0xa100
   1ddfe: 01 00        	c.nop
   1de00: 00 00 00 00  	v.nop
   1de04: 00 00 00 00 00 00    	m.nop
   1de0a: 2f 30 90 00  	l.ldi_hi r6, 0x200
   1de0e: 01 00        	c.nop
   1de10: 00 00 00 00  	v.nop
   1de14: 00 00 00 00 00 00    	m.nop
   1de1a: 65 29 06 02  	l.orix r5, r5, 0x6
   1de1e: 01 00        	c.nop
   1de20: 00 00 00 00  	v.nop
   1de24: 00 00 00 00 00 00    	m.nop
   1de2a: 2b 10 80 01  	l.jal r0, 0x2 <0x1de40>
   1de2e: 01 00        	c.nop
   1de30: 00 00 00 00  	v.nop
   1de34: 00 00 00 00 00 00    	m.nop
   1de3a: 00 00        	s.nop16
   1de3c: 00 00        	s.nop16
   1de3e: 81 42        	<unknown>
   1de40: 00 00 00 00  	v.nop
   1de44: 00 00 00 00 00 00    	m.nop
   1de4a: 2f 30 47 21  	l.ldi16.s r6, 0x8e0
   1de4e: 01 00        	c.nop
   1de50: 00 00 00 00  	v.nop
   1de54: 00 00 00 00 00 00    	m.nop
   1de5a: a3 11 06 18  	<unknown>
   1de5e: 01 00        	c.nop
   1de60: 00 00 00 00  	v.nop
   1de64: 00 00 00 00 00 00    	m.nop
   1de6a: b1 31 00 00  	l.ld.b r6, [r6]
   1de6e: 01 00        	c.nop
   1de70: 00 00 00 00  	v.nop
   1de74: 00 00 00 00 00 00    	m.nop
   1de7a: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1de7e: 01 00        	c.nop
   1de80: 00 00 00 00  	v.nop
   1de84: 00 00 00 00 00 00    	m.nop
   1de8a: 7b 35 00 00  	<unknown>
   1de8e: 01 00        	c.nop
   1de90: 00 00 00 00  	v.nop
   1de94: 00 00 00 00 00 00    	m.nop
   1de9a: 2f 37 46 21  	l.ldi16.s r6, 0x8dc
   1de9e: 01 00        	c.nop
   1dea0: 00 00 00 00  	v.nop
   1dea4: 00 00 00 00 00 00    	m.nop
   1deaa: a3 11 06 18  	<unknown>
   1deae: 01 00        	c.nop
   1deb0: 00 00 00 00  	v.nop
   1deb4: 00 00 00 00 00 00    	m.nop
   1deba: 2f 3e 46 21  	l.ldi16.s r7, 0x8d8
   1debe: 01 00        	c.nop
   1dec0: 00 00 00 00  	v.nop
   1dec4: 00 00 00 00 00 00    	m.nop
   1deca: e3 11 07 18  	<unknown>
   1dece: 01 00        	c.nop
   1ded0: 00 00 00 00  	v.nop
   1ded4: 00 00 00 00 00 00    	m.nop
   1deda: b1 31 40 00  	l.ld.w r6, [r6]
   1dede: 01 00        	c.nop
   1dee0: 00 00 00 00  	v.nop
   1dee4: 00 00 00 00 00 00    	m.nop
   1deea: f1 39 40 00  	l.ld.w r7, [r7]
   1deee: 01 00        	c.nop
   1def0: 00 00 00 00  	v.nop
   1def4: 00 00 00 00 00 00    	m.nop
   1defa: bb 37 07 40  	<unknown>
   1defe: 01 00        	c.nop
   1df00: 00 00 00 00  	v.nop
   1df04: 00 00 00 00 00 00    	m.nop
   1df0a: 2f 37 45 21  	l.ldi16.s r6, 0x8bc
   1df0e: 01 00        	c.nop
   1df10: 00 00 00 00  	v.nop
   1df14: 00 00 00 00 00 00    	m.nop
   1df1a: a3 11 06 18  	<unknown>
   1df1e: 01 00        	c.nop
   1df20: 00 00 00 00  	v.nop
   1df24: 00 00 00 00 00 00    	m.nop
   1df2a: 2f 3e 45 21  	l.ldi16.s r7, 0x8b8
   1df2e: 01 00        	c.nop
   1df30: 00 00 00 00  	v.nop
   1df34: 00 00 00 00 00 00    	m.nop
   1df3a: e3 11 07 18  	<unknown>
   1df3e: 01 00        	c.nop
   1df40: 00 00 00 00  	v.nop
   1df44: 00 00 00 00 00 00    	m.nop
   1df4a: b1 31 40 00  	l.ld.w r6, [r6]
   1df4e: 01 00        	c.nop
   1df50: 00 00 00 00  	v.nop
   1df54: 00 00 00 00 00 00    	m.nop
   1df5a: f1 39 40 00  	l.ld.w r7, [r7]
   1df5e: 01 00        	c.nop
   1df60: 00 00 00 00  	v.nop
   1df64: 00 00 00 00 00 00    	m.nop
   1df6a: 2f 47 46 21  	l.ldi16.s r8, 0x8dc
   1df6e: 01 00        	c.nop
   1df70: 00 00 00 00  	v.nop
   1df74: 00 00 00 00 00 00    	m.nop
   1df7a: 23 12 08 18  	<unknown>
   1df7e: 01 00        	c.nop
   1df80: 00 00 00 00  	v.nop
   1df84: 00 00 00 00 00 00    	m.nop
   1df8a: 31 40 cf 00  	l.st.w r15, [r8]
   1df8e: 01 00        	c.nop
   1df90: 00 00 00 00  	v.nop
   1df94: 00 00 00 00 00 00    	m.nop
   1df9a: 2f 3f 45 21  	l.ldi16.s r7, 0x8bc
   1df9e: 1e 93        	<unknown>
   1dfa0: 00 00 00 00  	v.nop
   1dfa4: 00 00 00 00 00 00    	m.nop
   1dfaa: e3 11 07 18  	<unknown>
   1dfae: 01 00        	c.nop
   1dfb0: 00 00 00 00  	v.nop
   1dfb4: 00 00 00 00 00 00    	m.nop
   1dfba: 31 38 c6 00  	l.st.w r6, [r7]
   1dfbe: 01 00        	c.nop
   1dfc0: 00 00 00 00  	v.nop
   1dfc4: 00 00 00 00 00 00    	m.nop
   1dfca: 2b 10 82 01  	l.jal r0, 0x22 <0x1e1e0>
   1dfce: 01 00        	c.nop
   1dfd0: 00 00 00 00  	v.nop
   1dfd4: 00 00 00 00 00 00    	m.nop
   1dfda: a7 11 a3 09  	l.addi_sft.s.sft0 r6, sp, 0x23
   1dfde: 01 00        	c.nop
   1dfe0: 00 00 00 00  	v.nop
   1dfe4: 00 00 00 00 00 00    	m.nop
   1dfea: b1 31 00 00  	l.ld.b r6, [r6]
   1dfee: 01 00        	c.nop
   1dff0: 00 00 00 00  	v.nop
   1dff4: 00 00 00 00 00 00    	m.nop
   1dffa: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1dffe: 01 00        	c.nop
   1e000: 00 00 00 00  	v.nop
   1e004: 00 00 00 00 00 00    	m.nop
   1e00a: 3b 31 60 00  	<unknown>
   1e00e: 01 00        	c.nop
   1e010: 00 00 00 00  	v.nop
   1e014: 00 00 00 00 00 00    	m.nop
   1e01a: 6f 32 45 21  	l.ldi16.s r6, 0x8a9
   1e01e: 01 00        	c.nop
   1e020: 00 00 00 00  	v.nop
   1e024: 00 00 00 00 00 00    	m.nop
   1e02a: a3 11 06 18  	<unknown>
   1e02e: 01 00        	c.nop
   1e030: 00 00 00 00  	v.nop
   1e034: 00 00 00 00 00 00    	m.nop
   1e03a: b1 31 00 00  	l.ld.b r6, [r6]
   1e03e: 01 00        	c.nop
   1e040: 00 00 00 00  	v.nop
   1e044: 00 00 00 00 00 00    	m.nop
   1e04a: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1e04e: 01 00        	c.nop
   1e050: 00 00 00 00  	v.nop
   1e054: 00 00 00 00 00 00    	m.nop
   1e05a: fb 36 20 00  	<unknown>
   1e05e: 01 00        	c.nop
   1e060: 00 00 00 00  	v.nop
   1e064: 00 00 00 00 00 00    	m.nop
   1e06a: 31 12 40 08  	l.ld.w r8, [sp, 0x80]
   1e06e: 01 00        	c.nop
   1e070: 00 00 00 00  	v.nop
   1e074: 00 00 00 00 00 00    	m.nop
   1e07a: b1 41 40 00  	l.ld.w r6, [r8]
   1e07e: 01 00        	c.nop
   1e080: 00 00 00 00  	v.nop
   1e084: 00 00 00 00 00 00    	m.nop
   1e08a: f1 41 44 00  	l.ld.w r7, [r8, 0x4]
   1e08e: 01 00        	c.nop
   1e090: 00 00 00 00  	v.nop
   1e094: 00 00 00 00 00 00    	m.nop
   1e09a: 31 42 48 00  	l.ld.w r8, [r8, 0x8]
   1e09e: 01 00        	c.nop
   1e0a0: 00 00 00 00  	v.nop
   1e0a4: 00 00 00 00 00 00    	m.nop
   1e0aa: a3 39 06 19  	<unknown>
   1e0ae: 01 00        	c.nop
   1e0b0: 00 00 00 00  	v.nop
   1e0b4: 00 00 00 00 00 00    	m.nop
   1e0ba: a3 31 08 19  	<unknown>
   1e0be: 01 00        	c.nop
   1e0c0: 00 00 00 00  	v.nop
   1e0c4: 00 00 00 00 00 00    	m.nop
   1e0ca: 2b 28 83 01  	l.jal r0, 0x35 <0x1e410>
   1e0ce: 01 00        	c.nop
   1e0d0: 00 00 00 00  	v.nop
   1e0d4: 00 00 00 00 00 00    	m.nop
   1e0da: bb 33 00 00  	<unknown>
   1e0de: 01 00        	c.nop
   1e0e0: 00 00 00 00  	v.nop
   1e0e4: 00 00 00 00 00 00    	m.nop
   1e0ea: 2f 34 44 21  	l.ldi16.s r6, 0x890
   1e0ee: 01 00        	c.nop
   1e0f0: 00 00 00 00  	v.nop
   1e0f4: 00 00 00 00 00 00    	m.nop
   1e0fa: a3 11 06 18  	<unknown>
   1e0fe: 01 00        	c.nop
   1e100: 00 00 00 00  	v.nop
   1e104: 00 00 00 00 00 00    	m.nop
   1e10a: 31 32 40 00  	l.ld.w r8, [r6]
   1e10e: 01 00        	c.nop
   1e110: 00 00 00 00  	v.nop
   1e114: 00 00 00 00 00 00    	m.nop
   1e11a: 2f 37 44 21  	l.ldi16.s r6, 0x89c
   1e11e: 01 00        	c.nop
   1e120: 00 00 00 00  	v.nop
   1e124: 00 00 00 00 00 00    	m.nop
   1e12a: a3 11 06 18  	<unknown>
   1e12e: 01 00        	c.nop
   1e130: 00 00 00 00  	v.nop
   1e134: 00 00 00 00 00 00    	m.nop
   1e13a: b1 31 40 00  	l.ld.w r6, [r6]
   1e13e: 01 00        	c.nop
   1e140: 00 00 00 00  	v.nop
   1e144: 00 00 00 00 00 00    	m.nop
   1e14a: 2f 38 08 29  	l.ldi16.s r7, 0x4100
   1e14e: 01 00        	c.nop
   1e150: 00 00 00 00  	v.nop
   1e154: 00 00 00 00 00 00    	m.nop
   1e15a: af 3f 91 08  	l.ldi_hi r7, 0x423e
   1e15e: 01 00        	c.nop
   1e160: 00 00 00 00  	v.nop
   1e164: 00 00 00 00 00 00    	m.nop
   1e16a: fb 40 0f 00  	<unknown>
   1e16e: 01 00        	c.nop
   1e170: 00 00 00 00  	v.nop
   1e174: 00 00 00 00 00 00    	m.nop
   1e17a: 2f 38 08 15  	l.ldi16.u r7, 0xa100
   1e17e: 01 00        	c.nop
   1e180: 00 00 00 00  	v.nop
   1e184: 00 00 00 00 00 00    	m.nop
   1e18a: 2f 38 80 08  	l.ldi_hi r7, 0x4000
   1e18e: 01 00        	c.nop
   1e190: 00 00 00 00  	v.nop
   1e194: 00 00 00 00 00 00    	m.nop
   1e19a: 31 38 c6 00  	l.st.w r6, [r7]
   1e19e: 01 00        	c.nop
   1e1a0: 00 00 00 00  	v.nop
   1e1a4: 00 00 00 00 00 00    	m.nop
   1e1aa: 2b 70 81 01  	l.jal r0, 0x1e <0x1e380>
   1e1ae: 01 00        	c.nop
   1e1b0: 00 00 00 00  	v.nop
   1e1b4: 00 00 00 00 00 00    	m.nop
   1e1ba: 2f 37 45 21  	l.ldi16.s r6, 0x8bc
   1e1be: 01 00        	c.nop
   1e1c0: 00 00 00 00  	v.nop
   1e1c4: 00 00 00 00 00 00    	m.nop
   1e1ca: a3 11 06 18  	<unknown>
   1e1ce: 01 00        	c.nop
   1e1d0: 00 00 00 00  	v.nop
   1e1d4: 00 00 00 00 00 00    	m.nop
   1e1da: b1 31 40 00  	l.ld.w r6, [r6]
   1e1de: 01 00        	c.nop
   1e1e0: 00 00 00 00  	v.nop
   1e1e4: 00 00 00 00 00 00    	m.nop
   1e1ea: 2f 3c 45 21  	l.ldi16.s r7, 0x8b0
   1e1ee: 01 00        	c.nop
   1e1f0: 00 00 00 00  	v.nop
   1e1f4: 00 00 00 00 00 00    	m.nop
   1e1fa: e3 11 07 18  	<unknown>
   1e1fe: 01 00        	c.nop
   1e200: 00 00 00 00  	v.nop
   1e204: 00 00 00 00 00 00    	m.nop
   1e20a: 2f 45 45 21  	l.ldi16.s r8, 0x8b4
   1e20e: 01 00        	c.nop
   1e210: 00 00 00 00  	v.nop
   1e214: 00 00 00 00 00 00    	m.nop
   1e21a: 23 12 08 18  	<unknown>
   1e21e: 01 00        	c.nop
   1e220: 00 00 00 00  	v.nop
   1e224: 00 00 00 00 00 00    	m.nop
   1e22a: 31 42 40 00  	l.ld.w r8, [r8]
   1e22e: 01 00        	c.nop
   1e230: 00 00 00 00  	v.nop
   1e234: 00 00 00 00 00 00    	m.nop
   1e23a: f1 39 40 00  	l.ld.w r7, [r7]
   1e23e: 01 00        	c.nop
   1e240: 00 00 00 00  	v.nop
   1e244: 00 00 00 00 00 00    	m.nop
   1e24a: ef 57 7f 1f  	l.ldi16.u r10, 0xffff
   1e24e: 01 00        	c.nop
   1e250: 00 00 00 00  	v.nop
   1e254: 00 00 00 00 00 00    	m.nop
   1e25a: 2f 4f 44 21  	l.ldi16.s r9, 0x89c
   1e25e: 01 00        	c.nop
   1e260: 00 00 00 00  	v.nop
   1e264: 00 00 00 00 00 00    	m.nop
   1e26a: ef 57 83 00  	l.ldi_hi r10, 0x7f
   1e26e: 01 00        	c.nop
   1e270: 00 00 00 00  	v.nop
   1e274: 00 00 00 00 00 00    	m.nop
   1e27a: 63 12 09 18  	<unknown>
   1e27e: aa 85        	<unknown>
   1e280: 00 00 00 00  	v.nop
   1e284: 00 00 00 00 00 00    	m.nop
   1e28a: 2f 50 00 25  	l.ldi16.s r10, 0x2000
   1e28e: 12 04        	<unknown>
   1e290: 00 00 00 00  	v.nop
   1e294: 00 00 00 00 00 00    	m.nop
   1e29a: 6f 57 91 08  	l.ldi_hi r10, 0x423d
   1e29e: e6 03        	<unknown>
   1e2a0: 00 00 00 00  	v.nop
   1e2a4: 00 00 00 00 00 00    	m.nop
   1e2aa: 71 4a 40 00  	l.ld.w r9, [r9]
   1e2ae: 2a 94        	<unknown>
   1e2b0: 00 00 00 00  	v.nop
   1e2b4: 00 00 00 00 00 00    	m.nop
   1e2ba: 2f 56 33 2b  	l.ldi16.s r10, 0x5678
   1e2be: 01 00        	c.nop
   1e2c0: 00 00 00 00  	v.nop
   1e2c4: 00 00 00 00 00 00    	m.nop
   1e2ca: 2f 55 91 02  	l.ldi_hi r10, 0x1234
   1e2ce: 01 00        	c.nop
   1e2d0: 00 00 00 00  	v.nop
   1e2d4: 00 00 00 00 00 00    	m.nop
   1e2da: a5 31 8b 01  	l.andix r6, r6, 0xb
   1e2de: 01 00        	c.nop
   1e2e0: 00 00 00 00  	v.nop
   1e2e4: 00 00 00 00 00 00    	m.nop
   1e2ea: 31 40 ca 00  	l.st.w r10, [r8]
   1e2ee: 01 00        	c.nop
   1e2f0: 00 00 00 00  	v.nop
   1e2f4: 00 00 00 00 00 00    	m.nop
   1e2fa: a5 3a 0b 02  	l.orix r10, r7, 0xb
   1e2fe: 01 00        	c.nop
   1e300: 00 00 00 00  	v.nop
   1e304: 00 00 00 00 00 00    	m.nop
   1e30a: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   1e30e: 01 00        	c.nop
   1e310: 00 00 00 00  	v.nop
   1e314: 00 00 00 00 00 00    	m.nop
   1e31a: ef 5f ff 00  	l.ldi_hi r11, 0xfff
   1e31e: 01 00        	c.nop
   1e320: 00 00 00 00  	v.nop
   1e324: 00 00 00 00 00 00    	m.nop
   1e32a: a5 39 06 02  	l.orix r6, r7, 0x6
   1e32e: 01 00        	c.nop
   1e330: 00 00 00 00  	v.nop
   1e334: 00 00 00 00 00 00    	m.nop
   1e33a: 31 41 cb 00  	l.st.w r11, [r8, 0x4]
   1e33e: 01 00        	c.nop
   1e340: 00 00 00 00  	v.nop
   1e344: 00 00 00 00 00 00    	m.nop
   1e34a: 31 42 ca 00  	l.st.w r10, [r8, 0x8]
   1e34e: 01 00        	c.nop
   1e350: 00 00 00 00  	v.nop
   1e354: 00 00 00 00 00 00    	m.nop
   1e35a: 31 40 c9 00  	l.st.w r9, [r8]
   1e35e: 01 00        	c.nop
   1e360: 00 00 00 00  	v.nop
   1e364: 00 00 00 00 00 00    	m.nop
   1e36a: 31 41 c5 00  	l.st.w r5, [r8, 0x4]
   1e36e: 01 00        	c.nop
   1e370: 00 00 00 00  	v.nop
   1e374: 00 00 00 00 00 00    	m.nop
   1e37a: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   1e37e: 01 00        	c.nop
   1e380: 00 00 00 00  	v.nop
   1e384: 00 00 00 00 00 00    	m.nop
   1e38a: 2f 37 46 21  	l.ldi16.s r6, 0x8dc
   1e38e: 01 00        	c.nop
   1e390: 00 00 00 00  	v.nop
   1e394: 00 00 00 00 00 00    	m.nop
   1e39a: a3 11 06 18  	<unknown>
   1e39e: 01 00        	c.nop
   1e3a0: 00 00 00 00  	v.nop
   1e3a4: 00 00 00 00 00 00    	m.nop
   1e3aa: b1 31 40 00  	l.ld.w r6, [r6]
   1e3ae: 01 00        	c.nop
   1e3b0: 00 00 00 00  	v.nop
   1e3b4: 00 00 00 00 00 00    	m.nop
   1e3ba: 2f 3f 46 21  	l.ldi16.s r7, 0x8dc
   1e3be: 01 00        	c.nop
   1e3c0: 00 00 00 00  	v.nop
   1e3c4: 00 00 00 00 00 00    	m.nop
   1e3ca: e3 11 07 18  	<unknown>
   1e3ce: 01 00        	c.nop
   1e3d0: 00 00 00 00  	v.nop
   1e3d4: 00 00 00 00 00 00    	m.nop
   1e3da: 00 00        	s.nop16
   1e3dc: 00 00        	s.nop16
   1e3de: 05 03        	<unknown>
   1e3e0: 00 00 00 00  	v.nop
   1e3e4: 00 00 00 00 00 00    	m.nop
   1e3ea: 31 38 c6 00  	l.st.w r6, [r7]
   1e3ee: 01 00        	c.nop
   1e3f0: 00 00 00 00  	v.nop
   1e3f4: 00 00 00 00 00 00    	m.nop
   1e3fa: 2b 28 82 01  	l.jal r0, 0x25 <0x1e640>
   1e3fe: 01 00        	c.nop
   1e400: 00 00 00 00  	v.nop
   1e404: 00 00 00 00 00 00    	m.nop
   1e40a: 00 00        	s.nop16
   1e40c: 00 00        	s.nop16
   1e40e: 05 43        	<unknown>
   1e410: 00 00 00 00  	v.nop
   1e414: 00 00 00 00 00 00    	m.nop
   1e41a: 2f 3f 45 21  	l.ldi16.s r7, 0x8bc
   1e41e: 01 00        	c.nop
   1e420: 00 00 00 00  	v.nop
   1e424: 00 00 00 00 00 00    	m.nop
   1e42a: e3 11 07 18  	<unknown>
   1e42e: 01 00        	c.nop
   1e430: 00 00 00 00  	v.nop
   1e434: 00 00 00 00 00 00    	m.nop
   1e43a: 2f 44 45 21  	l.ldi16.s r8, 0x8b0
   1e43e: 01 00        	c.nop
   1e440: 00 00 00 00  	v.nop
   1e444: 00 00 00 00 00 00    	m.nop
   1e44a: 23 12 08 18  	<unknown>
   1e44e: 01 00        	c.nop
   1e450: 00 00 00 00  	v.nop
   1e454: 00 00 00 00 00 00    	m.nop
   1e45a: 2f 4d 45 21  	l.ldi16.s r9, 0x8b4
   1e45e: 01 00        	c.nop
   1e460: 00 00 00 00  	v.nop
   1e464: 00 00 00 00 00 00    	m.nop
   1e46a: 63 12 09 18  	<unknown>
   1e46e: 01 00        	c.nop
   1e470: 00 00 00 00  	v.nop
   1e474: 00 00 00 00 00 00    	m.nop
   1e47a: f1 39 40 00  	l.ld.w r7, [r7]
   1e47e: 01 00        	c.nop
   1e480: 00 00 00 00  	v.nop
   1e484: 00 00 00 00 00 00    	m.nop
   1e48a: 31 42 40 00  	l.ld.w r8, [r8]
   1e48e: 01 00        	c.nop
   1e490: 00 00 00 00  	v.nop
   1e494: 00 00 00 00 00 00    	m.nop
   1e49a: 71 4a 40 00  	l.ld.w r9, [r9]
   1e49e: 01 00        	c.nop
   1e4a0: 00 00 00 00  	v.nop
   1e4a4: 00 00 00 00 00 00    	m.nop
   1e4aa: 2f 57 44 21  	l.ldi16.s r10, 0x89c
   1e4ae: 01 00        	c.nop
   1e4b0: 00 00 00 00  	v.nop
   1e4b4: 00 00 00 00 00 00    	m.nop
   1e4ba: a3 12 0a 18  	<unknown>
   1e4be: 01 00        	c.nop
   1e4c0: 00 00 00 00  	v.nop
   1e4c4: 00 00 00 00 00 00    	m.nop
   1e4ca: a3 39 06 18  	<unknown>
   1e4ce: 01 00        	c.nop
   1e4d0: 00 00 00 00  	v.nop
   1e4d4: 00 00 00 00 00 00    	m.nop
   1e4da: e3 41 19 03  	l.slli r7, r8, 0x19
   1e4de: 01 00        	c.nop
   1e4e0: 00 00 00 00  	v.nop
   1e4e4: 00 00 00 00 00 00    	m.nop
   1e4ea: 23 4a 04 03  	l.slli r8, r9, 0x4
   1e4ee: 01 00        	c.nop
   1e4f0: 00 00 00 00  	v.nop
   1e4f4: 00 00 00 00 00 00    	m.nop
   1e4fa: ef 4f 7f 1f  	l.ldi16.u r9, 0xffff
   1e4fe: 01 00        	c.nop
   1e500: 00 00 00 00  	v.nop
   1e504: 00 00 00 00 00 00    	m.nop
   1e50a: 2f 60 00 25  	l.ldi16.s r12, 0x2000
   1e50e: 01 00        	c.nop
   1e510: 00 00 00 00  	v.nop
   1e514: 00 00 00 00 00 00    	m.nop
   1e51a: ef 4f 83 00  	l.ldi_hi r9, 0x7f
   1e51e: 01 00        	c.nop
   1e520: 00 00 00 00  	v.nop
   1e524: 00 00 00 00 00 00    	m.nop
   1e52a: 6f 67 91 08  	l.ldi_hi r12, 0x423d
   1e52e: a6 85        	<unknown>
   1e530: 00 00 00 00  	v.nop
   1e534: 00 00 00 00 00 00    	m.nop
   1e53a: 65 32 8b 01  	l.andix r9, r6, 0xb
   1e53e: 32 94        	<unknown>
   1e540: 00 00 00 00  	v.nop
   1e544: 00 00 00 00 00 00    	m.nop
   1e54a: 2f 67 45 21  	l.ldi16.s r12, 0x8bc
   1e54e: 01 00        	c.nop
   1e550: 00 00 00 00  	v.nop
   1e554: 00 00 00 00 00 00    	m.nop
   1e55a: 23 13 0c 18  	<unknown>
   1e55e: 01 00        	c.nop
   1e560: 00 00 00 00  	v.nop
   1e564: 00 00 00 00 00 00    	m.nop
   1e56a: 31 60 c6 00  	l.st.w r6, [r12]
   1e56e: 01 00        	c.nop
   1e570: 00 00 00 00  	v.nop
   1e574: 00 00 00 00 00 00    	m.nop
   1e57a: 2f 36 33 2b  	l.ldi16.s r6, 0x5678
   1e57e: 01 00        	c.nop
   1e580: 00 00 00 00  	v.nop
   1e584: 00 00 00 00 00 00    	m.nop
   1e58a: b1 52 40 00  	l.ld.w r10, [r10]
   1e58e: 01 00        	c.nop
   1e590: 00 00 00 00  	v.nop
   1e594: 00 00 00 00 00 00    	m.nop
   1e59a: 2f 35 91 02  	l.ldi_hi r6, 0x1234
   1e59e: 01 00        	c.nop
   1e5a0: 00 00 00 00  	v.nop
   1e5a4: 00 00 00 00 00 00    	m.nop
   1e5aa: 31 40 c6 00  	l.st.w r6, [r8]
   1e5ae: 01 00        	c.nop
   1e5b0: 00 00 00 00  	v.nop
   1e5b4: 00 00 00 00 00 00    	m.nop
   1e5ba: a5 39 0b 02  	l.orix r6, r7, 0xb
   1e5be: 01 00        	c.nop
   1e5c0: 00 00 00 00  	v.nop
   1e5c4: 00 00 00 00 00 00    	m.nop
   1e5ca: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   1e5ce: 01 00        	c.nop
   1e5d0: 00 00 00 00  	v.nop
   1e5d4: 00 00 00 00 00 00    	m.nop
   1e5da: ef 5f ff 00  	l.ldi_hi r11, 0xfff
   1e5de: 01 00        	c.nop
   1e5e0: 00 00 00 00  	v.nop
   1e5e4: 00 00 00 00 00 00    	m.nop
   1e5ea: 31 41 cb 00  	l.st.w r11, [r8, 0x4]
   1e5ee: 01 00        	c.nop
   1e5f0: 00 00 00 00  	v.nop
   1e5f4: 00 00 00 00 00 00    	m.nop
   1e5fa: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   1e5fe: 01 00        	c.nop
   1e600: 00 00 00 00  	v.nop
   1e604: 00 00 00 00 00 00    	m.nop
   1e60a: a5 39 09 02  	l.orix r6, r7, 0x9
   1e60e: 01 00        	c.nop
   1e610: 00 00 00 00  	v.nop
   1e614: 00 00 00 00 00 00    	m.nop
   1e61a: 31 40 ca 00  	l.st.w r10, [r8]
   1e61e: 01 00        	c.nop
   1e620: 00 00 00 00  	v.nop
   1e624: 00 00 00 00 00 00    	m.nop
   1e62a: 31 41 c5 00  	l.st.w r5, [r8, 0x4]
   1e62e: 01 00        	c.nop
   1e630: 00 00 00 00  	v.nop
   1e634: 00 00 00 00 00 00    	m.nop
   1e63a: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   1e63e: 01 00        	c.nop
   1e640: 00 00 00 00  	v.nop
   1e644: 00 00 00 00 00 00    	m.nop
   1e64a: 2f 33 47 21  	l.ldi16.s r6, 0x8ec
   1e64e: 01 00        	c.nop
   1e650: 00 00 00 00  	v.nop
   1e654: 00 00 00 00 00 00    	m.nop
   1e65a: a3 11 06 18  	<unknown>
   1e65e: 01 00        	c.nop
   1e660: 00 00 00 00  	v.nop
   1e664: 00 00 00 00 00 00    	m.nop
   1e66a: b1 31 00 00  	l.ld.b r6, [r6]
   1e66e: 01 00        	c.nop
   1e670: 00 00 00 00  	v.nop
   1e674: 00 00 00 00 00 00    	m.nop
   1e67a: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1e67e: 01 00        	c.nop
   1e680: 00 00 00 00  	v.nop
   1e684: 00 00 00 00 00 00    	m.nop
   1e68a: fb 31 40 00  	<unknown>
   1e68e: 01 00        	c.nop
   1e690: 00 00 00 00  	v.nop
   1e694: 00 00 00 00 00 00    	m.nop
   1e69a: 2f 32 47 21  	l.ldi16.s r6, 0x8e8
   1e69e: 01 00        	c.nop
   1e6a0: 00 00 00 00  	v.nop
   1e6a4: 00 00 00 00 00 00    	m.nop
   1e6aa: a3 11 06 18  	<unknown>
   1e6ae: 01 00        	c.nop
   1e6b0: 00 00 00 00  	v.nop
   1e6b4: 00 00 00 00 00 00    	m.nop
   1e6ba: 2f 39 47 21  	l.ldi16.s r7, 0x8e4
   1e6be: 01 00        	c.nop
   1e6c0: 00 00 00 00  	v.nop
   1e6c4: 00 00 00 00 00 00    	m.nop
   1e6ca: e3 11 07 18  	<unknown>
   1e6ce: 01 00        	c.nop
   1e6d0: 00 00 00 00  	v.nop
   1e6d4: 00 00 00 00 00 00    	m.nop
   1e6da: b1 31 40 00  	l.ld.w r6, [r6]
   1e6de: 01 00        	c.nop
   1e6e0: 00 00 00 00  	v.nop
   1e6e4: 00 00 00 00 00 00    	m.nop
   1e6ea: f1 39 40 00  	l.ld.w r7, [r7]
   1e6ee: 01 00        	c.nop
   1e6f0: 00 00 00 00  	v.nop
   1e6f4: 00 00 00 00 00 00    	m.nop
   1e6fa: bb 33 07 40  	<unknown>
   1e6fe: 01 00        	c.nop
   1e700: 00 00 00 00  	v.nop
   1e704: 00 00 00 00 00 00    	m.nop
   1e70a: 2f 35 46 21  	l.ldi16.s r6, 0x8d4
   1e70e: 01 00        	c.nop
   1e710: 00 00 00 00  	v.nop
   1e714: 00 00 00 00 00 00    	m.nop
   1e71a: a3 11 06 18  	<unknown>
   1e71e: 01 00        	c.nop
   1e720: 00 00 00 00  	v.nop
   1e724: 00 00 00 00 00 00    	m.nop
   1e72a: 2f 3c 46 21  	l.ldi16.s r7, 0x8d0
   1e72e: 01 00        	c.nop
   1e730: 00 00 00 00  	v.nop
   1e734: 00 00 00 00 00 00    	m.nop
   1e73a: e3 11 07 18  	<unknown>
   1e73e: 01 00        	c.nop
   1e740: 00 00 00 00  	v.nop
   1e744: 00 00 00 00 00 00    	m.nop
   1e74a: b1 31 40 00  	l.ld.w r6, [r6]
   1e74e: 01 00        	c.nop
   1e750: 00 00 00 00  	v.nop
   1e754: 00 00 00 00 00 00    	m.nop
   1e75a: f1 39 40 00  	l.ld.w r7, [r7]
   1e75e: 01 00        	c.nop
   1e760: 00 00 00 00  	v.nop
   1e764: 00 00 00 00 00 00    	m.nop
   1e76a: 2f 42 47 21  	l.ldi16.s r8, 0x8e8
   1e76e: 01 00        	c.nop
   1e770: 00 00 00 00  	v.nop
   1e774: 00 00 00 00 00 00    	m.nop
   1e77a: 23 12 08 18  	<unknown>
   1e77e: 01 00        	c.nop
   1e780: 00 00 00 00  	v.nop
   1e784: 00 00 00 00 00 00    	m.nop
   1e78a: 31 40 cf 00  	l.st.w r15, [r8]
   1e78e: 01 00        	c.nop
   1e790: 00 00 00 00  	v.nop
   1e794: 00 00 00 00 00 00    	m.nop
   1e79a: 2f 3d 46 21  	l.ldi16.s r7, 0x8d4
   1e79e: 1e 93        	<unknown>
   1e7a0: 00 00 00 00  	v.nop
   1e7a4: 00 00 00 00 00 00    	m.nop
   1e7aa: e3 11 07 18  	<unknown>
   1e7ae: 01 00        	c.nop
   1e7b0: 00 00 00 00  	v.nop
   1e7b4: 00 00 00 00 00 00    	m.nop
   1e7ba: 31 38 c6 00  	l.st.w r6, [r7]
   1e7be: 01 00        	c.nop
   1e7c0: 00 00 00 00  	v.nop
   1e7c4: 00 00 00 00 00 00    	m.nop
   1e7ca: 2b 10 81 01  	l.jal r0, 0x12 <0x1e8e0>
   1e7ce: 01 00        	c.nop
   1e7d0: 00 00 00 00  	v.nop
   1e7d4: 00 00 00 00 00 00    	m.nop
   1e7da: bb 33 00 00  	<unknown>
   1e7de: 01 00        	c.nop
   1e7e0: 00 00 00 00  	v.nop
   1e7e4: 00 00 00 00 00 00    	m.nop
   1e7ea: 2f 2c 44 21  	l.ldi16.s r5, 0x890
   1e7ee: 01 00        	c.nop
   1e7f0: 00 00 00 00  	v.nop
   1e7f4: 00 00 00 00 00 00    	m.nop
   1e7fa: 63 11 05 18  	<unknown>
   1e7fe: 01 00        	c.nop
   1e800: 00 00 00 00  	v.nop
   1e804: 00 00 00 00 00 00    	m.nop
   1e80a: f1 29 40 00  	l.ld.w r7, [r5]
   1e80e: 01 00        	c.nop
   1e810: 00 00 00 00  	v.nop
   1e814: 00 00 00 00 00 00    	m.nop
   1e81a: 2f 28 45 21  	l.ldi16.s r5, 0x8a0
   1e81e: 01 00        	c.nop
   1e820: 00 00 00 00  	v.nop
   1e824: 00 00 00 00 00 00    	m.nop
   1e82a: 63 11 05 18  	<unknown>
   1e82e: 01 00        	c.nop
   1e830: 00 00 00 00  	v.nop
   1e834: 00 00 00 00 00 00    	m.nop
   1e83a: 71 29 40 00  	l.ld.w r5, [r5]
   1e83e: 01 00        	c.nop
   1e840: 00 00 00 00  	v.nop
   1e844: 00 00 00 00 00 00    	m.nop
   1e84a: 2f 30 08 29  	l.ldi16.s r6, 0x4100
   1e84e: 01 00        	c.nop
   1e850: 00 00 00 00  	v.nop
   1e854: 00 00 00 00 00 00    	m.nop
   1e85a: af 37 91 08  	l.ldi_hi r6, 0x423e
   1e85e: 01 00        	c.nop
   1e860: 00 00 00 00  	v.nop
   1e864: 00 00 00 00 00 00    	m.nop
   1e86a: fb 38 0f 00  	<unknown>
   1e86e: 01 00        	c.nop
   1e870: 00 00 00 00  	v.nop
   1e874: 00 00 00 00 00 00    	m.nop
   1e87a: 2f 30 08 15  	l.ldi16.u r6, 0xa100
   1e87e: 01 00        	c.nop
   1e880: 00 00 00 00  	v.nop
   1e884: 00 00 00 00 00 00    	m.nop
   1e88a: 2f 30 80 08  	l.ldi_hi r6, 0x4000
   1e88e: 01 00        	c.nop
   1e890: 00 00 00 00  	v.nop
   1e894: 00 00 00 00 00 00    	m.nop
   1e89a: 31 30 c5 00  	l.st.w r5, [r6]
   1e89e: 01 00        	c.nop
   1e8a0: 00 00 00 00  	v.nop
   1e8a4: 00 00 00 00 00 00    	m.nop
   1e8aa: 2b 70 81 01  	l.jal r0, 0x1e <0x1ea80>
   1e8ae: 01 00        	c.nop
   1e8b0: 00 00 00 00  	v.nop
   1e8b4: 00 00 00 00 00 00    	m.nop
   1e8ba: 2f 35 46 21  	l.ldi16.s r6, 0x8d4
   1e8be: 01 00        	c.nop
   1e8c0: 00 00 00 00  	v.nop
   1e8c4: 00 00 00 00 00 00    	m.nop
   1e8ca: a3 11 06 18  	<unknown>
   1e8ce: 01 00        	c.nop
   1e8d0: 00 00 00 00  	v.nop
   1e8d4: 00 00 00 00 00 00    	m.nop
   1e8da: b1 31 40 00  	l.ld.w r6, [r6]
   1e8de: 01 00        	c.nop
   1e8e0: 00 00 00 00  	v.nop
   1e8e4: 00 00 00 00 00 00    	m.nop
   1e8ea: 2f 3a 46 21  	l.ldi16.s r7, 0x8c8
   1e8ee: 01 00        	c.nop
   1e8f0: 00 00 00 00  	v.nop
   1e8f4: 00 00 00 00 00 00    	m.nop
   1e8fa: e3 11 07 18  	<unknown>
   1e8fe: 01 00        	c.nop
   1e900: 00 00 00 00  	v.nop
   1e904: 00 00 00 00 00 00    	m.nop
   1e90a: 2f 43 46 21  	l.ldi16.s r8, 0x8cc
   1e90e: 01 00        	c.nop
   1e910: 00 00 00 00  	v.nop
   1e914: 00 00 00 00 00 00    	m.nop
   1e91a: 23 12 08 18  	<unknown>
   1e91e: 01 00        	c.nop
   1e920: 00 00 00 00  	v.nop
   1e924: 00 00 00 00 00 00    	m.nop
   1e92a: 31 42 40 00  	l.ld.w r8, [r8]
   1e92e: 01 00        	c.nop
   1e930: 00 00 00 00  	v.nop
   1e934: 00 00 00 00 00 00    	m.nop
   1e93a: f1 39 40 00  	l.ld.w r7, [r7]
   1e93e: 01 00        	c.nop
   1e940: 00 00 00 00  	v.nop
   1e944: 00 00 00 00 00 00    	m.nop
   1e94a: ef 57 7f 1f  	l.ldi16.u r10, 0xffff
   1e94e: 01 00        	c.nop
   1e950: 00 00 00 00  	v.nop
   1e954: 00 00 00 00 00 00    	m.nop
   1e95a: 2f 48 45 21  	l.ldi16.s r9, 0x8a0
   1e95e: 01 00        	c.nop
   1e960: 00 00 00 00  	v.nop
   1e964: 00 00 00 00 00 00    	m.nop
   1e96a: ef 57 83 00  	l.ldi_hi r10, 0x7f
   1e96e: 01 00        	c.nop
   1e970: 00 00 00 00  	v.nop
   1e974: 00 00 00 00 00 00    	m.nop
   1e97a: 63 12 09 18  	<unknown>
   1e97e: aa 85        	<unknown>
   1e980: 00 00 00 00  	v.nop
   1e984: 00 00 00 00 00 00    	m.nop
   1e98a: 2f 50 00 25  	l.ldi16.s r10, 0x2000
   1e98e: 12 04        	<unknown>
   1e990: 00 00 00 00  	v.nop
   1e994: 00 00 00 00 00 00    	m.nop
   1e99a: 6f 57 91 08  	l.ldi_hi r10, 0x423d
   1e99e: e6 03        	<unknown>
   1e9a0: 00 00 00 00  	v.nop
   1e9a4: 00 00 00 00 00 00    	m.nop
   1e9aa: 71 4a 40 00  	l.ld.w r9, [r9]
   1e9ae: 2a 94        	<unknown>
   1e9b0: 00 00 00 00  	v.nop
   1e9b4: 00 00 00 00 00 00    	m.nop
   1e9ba: 2f 56 33 2b  	l.ldi16.s r10, 0x5678
   1e9be: 01 00        	c.nop
   1e9c0: 00 00 00 00  	v.nop
   1e9c4: 00 00 00 00 00 00    	m.nop
   1e9ca: 2f 55 91 02  	l.ldi_hi r10, 0x1234
   1e9ce: 01 00        	c.nop
   1e9d0: 00 00 00 00  	v.nop
   1e9d4: 00 00 00 00 00 00    	m.nop
   1e9da: a5 31 8b 01  	l.andix r6, r6, 0xb
   1e9de: 01 00        	c.nop
   1e9e0: 00 00 00 00  	v.nop
   1e9e4: 00 00 00 00 00 00    	m.nop
   1e9ea: 31 40 ca 00  	l.st.w r10, [r8]
   1e9ee: 01 00        	c.nop
   1e9f0: 00 00 00 00  	v.nop
   1e9f4: 00 00 00 00 00 00    	m.nop
   1e9fa: a5 3a 0b 02  	l.orix r10, r7, 0xb
   1e9fe: 01 00        	c.nop
   1ea00: 00 00 00 00  	v.nop
   1ea04: 00 00 00 00 00 00    	m.nop
   1ea0a: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   1ea0e: 01 00        	c.nop
   1ea10: 00 00 00 00  	v.nop
   1ea14: 00 00 00 00 00 00    	m.nop
   1ea1a: ef 5f ff 00  	l.ldi_hi r11, 0xfff
   1ea1e: 01 00        	c.nop
   1ea20: 00 00 00 00  	v.nop
   1ea24: 00 00 00 00 00 00    	m.nop
   1ea2a: a5 39 06 02  	l.orix r6, r7, 0x6
   1ea2e: 01 00        	c.nop
   1ea30: 00 00 00 00  	v.nop
   1ea34: 00 00 00 00 00 00    	m.nop
   1ea3a: 31 41 cb 00  	l.st.w r11, [r8, 0x4]
   1ea3e: 01 00        	c.nop
   1ea40: 00 00 00 00  	v.nop
   1ea44: 00 00 00 00 00 00    	m.nop
   1ea4a: 31 42 ca 00  	l.st.w r10, [r8, 0x8]
   1ea4e: 01 00        	c.nop
   1ea50: 00 00 00 00  	v.nop
   1ea54: 00 00 00 00 00 00    	m.nop
   1ea5a: 31 40 c9 00  	l.st.w r9, [r8]
   1ea5e: 01 00        	c.nop
   1ea60: 00 00 00 00  	v.nop
   1ea64: 00 00 00 00 00 00    	m.nop
   1ea6a: 31 41 c5 00  	l.st.w r5, [r8, 0x4]
   1ea6e: 01 00        	c.nop
   1ea70: 00 00 00 00  	v.nop
   1ea74: 00 00 00 00 00 00    	m.nop
   1ea7a: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   1ea7e: 01 00        	c.nop
   1ea80: 00 00 00 00  	v.nop
   1ea84: 00 00 00 00 00 00    	m.nop
   1ea8a: 2f 2a 47 21  	l.ldi16.s r5, 0x8e8
   1ea8e: 01 00        	c.nop
   1ea90: 00 00 00 00  	v.nop
   1ea94: 00 00 00 00 00 00    	m.nop
   1ea9a: 63 11 05 18  	<unknown>
   1ea9e: 01 00        	c.nop
   1eaa0: 00 00 00 00  	v.nop
   1eaa4: 00 00 00 00 00 00    	m.nop
   1eaaa: 71 29 40 00  	l.ld.w r5, [r5]
   1eaae: 01 00        	c.nop
   1eab0: 00 00 00 00  	v.nop
   1eab4: 00 00 00 00 00 00    	m.nop
   1eaba: 2f 32 47 21  	l.ldi16.s r6, 0x8e8
   1eabe: 01 00        	c.nop
   1eac0: 00 00 00 00  	v.nop
   1eac4: 00 00 00 00 00 00    	m.nop
   1eaca: a3 11 06 18  	<unknown>
   1eace: 01 00        	c.nop
   1ead0: 00 00 00 00  	v.nop
   1ead4: 00 00 00 00 00 00    	m.nop
   1eada: 00 00        	s.nop16
   1eadc: 00 00        	s.nop16
   1eade: 85 02        	<unknown>
   1eae0: 00 00 00 00  	v.nop
   1eae4: 00 00 00 00 00 00    	m.nop
   1eaea: 31 30 c5 00  	l.st.w r5, [r6]
   1eaee: 01 00        	c.nop
   1eaf0: 00 00 00 00  	v.nop
   1eaf4: 00 00 00 00 00 00    	m.nop
   1eafa: 2f 2d 01 21  	l.ldi16.s r5, 0x34
   1eafe: 01 00        	c.nop
   1eb00: 00 00 00 00  	v.nop
   1eb04: 00 00 00 00 00 00    	m.nop
   1eb0a: 00 00        	s.nop16
   1eb0c: 00 00        	s.nop16
   1eb0e: 16 92        	<unknown>
   1eb10: 00 00 00 00  	v.nop
   1eb14: 00 00 00 00 00 00    	m.nop
   1eb1a: 31 20 da 00  	l.st.w r26, [r4]
   1eb1e: 01 00        	c.nop
   1eb20: 00 00 00 00  	v.nop
   1eb24: 00 00 00 00 00 00    	m.nop
   1eb2a: 31 11 5c 12  	l.ld.w r4, [sp, 0x13c]
   1eb2e: 01 00        	c.nop
   1eb30: 00 00 00 00  	v.nop
   1eb34: 00 00 00 00 00 00    	m.nop
   1eb3a: 71 11 50 36  	l.ld.w r5, [sp, 0x370]
   1eb3e: 01 00        	c.nop
   1eb40: 00 00 00 00  	v.nop
   1eb44: 00 00 00 00 00 00    	m.nop
   1eb4a: 31 15 cf 34  	l.st.w r15, [sp, 0x354]
   1eb4e: 01 00        	c.nop
   1eb50: 00 00 00 00  	v.nop
   1eb54: 00 00 00 00 00 00    	m.nop
   1eb5a: 31 14 c4 34  	l.st.w r4, [sp, 0x350]
   1eb5e: 01 00        	c.nop
   1eb60: 00 00 00 00  	v.nop
   1eb64: 00 00 00 00 00 00    	m.nop
   1eb6a: 31 11 58 3e  	l.ld.w r4, [sp, 0x3f8]
   1eb6e: 01 00        	c.nop
   1eb70: 00 00 00 00  	v.nop
   1eb74: 00 00 00 00 00 00    	m.nop
   1eb7a: bb 29 3a 00  	<unknown>
   1eb7e: 01 00        	c.nop
   1eb80: 00 00 00 00  	v.nop
   1eb84: 00 00 00 00 00 00    	m.nop
   1eb8a: 7b 2d 00 40  	<unknown>
   1eb8e: 01 00        	c.nop
   1eb90: 00 00 00 00  	v.nop
   1eb94: 00 00 00 00 00 00    	m.nop
   1eb9a: 73 b9 83 00  	l.movs r5, md23_wrk
   1eb9e: 01 00        	c.nop
   1eba0: 00 00 00 00  	v.nop
   1eba4: 00 00 00 00 00 00    	m.nop
   1ebaa: 65 29 96 01  	l.andix r5, r5, 0x16
   1ebae: 01 00        	c.nop
   1ebb0: 00 00 00 00  	v.nop
   1ebb4: 00 00 00 00 00 00    	m.nop
   1ebba: 7b 2c 00 00  	<unknown>
   1ebbe: 01 00        	c.nop
   1ebc0: 00 00 00 00  	v.nop
   1ebc4: 00 00 00 00 00 00    	m.nop
   1ebca: 2f 2a 04 21  	l.ldi16.s r5, 0x88
   1ebce: 01 00        	c.nop
   1ebd0: 00 00 00 00  	v.nop
   1ebd4: 00 00 00 00 00 00    	m.nop
   1ebda: 63 21 05 10  	<unknown>
   1ebde: 01 00        	c.nop
   1ebe0: 00 00 00 00  	v.nop
   1ebe4: 00 00 00 00 00 00    	m.nop
   1ebea: b1 29 40 00  	l.ld.w r6, [r5]
   1ebee: 01 00        	c.nop
   1ebf0: 00 00 00 00  	v.nop
   1ebf4: 00 00 00 00 00 00    	m.nop
   1ebfa: f3 a1 83 00  	l.movs r7, md20_wrk
   1ebfe: 01 00        	c.nop
   1ec00: 00 00 00 00  	v.nop
   1ec04: 00 00 00 00 00 00    	m.nop
   1ec0a: 2f 48 f0 3e  	l.ldi_hi.m1 r9, 0xfe00
   1ec0e: 86 03        	<unknown>
   1ec10: 00 00 00 00  	v.nop
   1ec14: 00 00 00 00 00 00    	m.nop
   1ec1a: e5 39 89 01  	l.andix r7, r7, 0x9
   1ec1e: 01 00        	c.nop
   1ec20: 00 00 00 00  	v.nop
   1ec24: 00 00 00 00 00 00    	m.nop
   1ec2a: ef 4f 7f 1f  	l.ldi16.u r9, 0xffff
   1ec2e: 01 00        	c.nop
   1ec30: 00 00 00 00  	v.nop
   1ec34: 00 00 00 00 00 00    	m.nop
   1ec3a: 33 02 80 00  	l.movs r8, sip_id
   1ec3e: 01 00        	c.nop
   1ec40: 00 00 00 00  	v.nop
   1ec44: 00 00 00 00 00 00    	m.nop
   1ec4a: 6f 48 80 00  	l.ldi_hi r9, 0x1
   1ec4e: 3e 04        	<unknown>
   1ec50: 00 00 00 00  	v.nop
   1ec54: 00 00 00 00 00 00    	m.nop
   1ec5a: a5 31 89 01  	l.andix r6, r6, 0x9
   1ec5e: 01 00        	c.nop
   1ec60: 00 00 00 00  	v.nop
   1ec64: 00 00 00 00 00 00    	m.nop
   1ec6a: 71 12 4c 02  	l.ld.w r9, [sp, 0x2c]
   1ec6e: 01 00        	c.nop
   1ec70: 00 00 00 00  	v.nop
   1ec74: 00 00 00 00 00 00    	m.nop
   1ec7a: a5 31 07 02  	l.orix r6, r6, 0x7
   1ec7e: 01 00        	c.nop
   1ec80: 00 00 00 00  	v.nop
   1ec84: 00 00 00 00 00 00    	m.nop
   1ec8a: 00 00        	s.nop16
   1ec8c: 00 00        	s.nop16
   1ec8e: 65 8c        	<unknown>
   1ec90: 00 00 00 00  	v.nop
   1ec94: 00 00 00 00 00 00    	m.nop
   1ec9a: a5 31 08 02  	l.orix r6, r6, 0x8
   1ec9e: 01 00        	c.nop
   1eca0: 00 00 00 00  	v.nop
   1eca4: 00 00 00 00 00 00    	m.nop
   1ecaa: a5 31 1d 02  	l.orix r6, r6, 0x1d
   1ecae: 01 00        	c.nop
   1ecb0: 00 00 00 00  	v.nop
   1ecb4: 00 00 00 00 00 00    	m.nop
   1ecba: 31 28 c6 00  	l.st.w r6, [r5]
   1ecbe: 01 00        	c.nop
   1ecc0: 00 00 00 00  	v.nop
   1ecc4: 00 00 00 00 00 00    	m.nop
   1ecca: 71 11 50 36  	l.ld.w r5, [sp, 0x370]
   1ecce: 01 00        	c.nop
   1ecd0: 00 00 00 00  	v.nop
   1ecd4: 00 00 00 00 00 00    	m.nop
   1ecda: 3b 2d 1a 00  	<unknown>
   1ecde: 01 00        	c.nop
   1ece0: 00 00 00 00  	v.nop
   1ece4: 00 00 00 00 00 00    	m.nop
   1ecea: bb 2a 20 40  	<unknown>
   1ecee: 01 00        	c.nop
   1ecf0: 00 00 00 00  	v.nop
   1ecf4: 00 00 00 00 00 00    	m.nop
   1ecfa: 2f 28 08 29  	l.ldi16.s r5, 0x4100
   1ecfe: 01 00        	c.nop
   1ed00: 00 00 00 00  	v.nop
   1ed04: 00 00 00 00 00 00    	m.nop
   1ed0a: af 2f 91 00  	l.ldi_hi r5, 0x23e
   1ed0e: 01 00        	c.nop
   1ed10: 00 00 00 00  	v.nop
   1ed14: 00 00 00 00 00 00    	m.nop
   1ed1a: b1 11 00 3c  	l.ld.b r6, [sp, 0x3c0]
   1ed1e: 01 00        	c.nop
   1ed20: 00 00 00 00  	v.nop
   1ed24: 00 00 00 00 00 00    	m.nop
   1ed2a: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1ed2e: 01 00        	c.nop
   1ed30: 00 00 00 00  	v.nop
   1ed34: 00 00 00 00 00 00    	m.nop
   1ed3a: fb 35 00 00  	<unknown>
   1ed3e: 01 00        	c.nop
   1ed40: 00 00 00 00  	v.nop
   1ed44: 00 00 00 00 00 00    	m.nop
   1ed4a: b1 11 5c 3a  	l.ld.w r6, [sp, 0x3bc]
   1ed4e: 01 00        	c.nop
   1ed50: 00 00 00 00  	v.nop
   1ed54: 00 00 00 00 00 00    	m.nop
   1ed5a: f1 11 58 3a  	l.ld.w r7, [sp, 0x3b8]
   1ed5e: 01 00        	c.nop
   1ed60: 00 00 00 00  	v.nop
   1ed64: 00 00 00 00 00 00    	m.nop
   1ed6a: fb 31 27 40  	<unknown>
   1ed6e: 01 00        	c.nop
   1ed70: 00 00 00 00  	v.nop
   1ed74: 00 00 00 00 00 00    	m.nop
   1ed7a: b1 11 5c 38  	l.ld.w r6, [sp, 0x39c]
   1ed7e: 01 00        	c.nop
   1ed80: 00 00 00 00  	v.nop
   1ed84: 00 00 00 00 00 00    	m.nop
   1ed8a: f1 11 58 38  	l.ld.w r7, [sp, 0x398]
   1ed8e: 01 00        	c.nop
   1ed90: 00 00 00 00  	v.nop
   1ed94: 00 00 00 00 00 00    	m.nop
   1ed9a: 31 17 cf 3a  	l.st.w r15, [sp, 0x3bc]
   1ed9e: 01 00        	c.nop
   1eda0: 00 00 00 00  	v.nop
   1eda4: 00 00 00 00 00 00    	m.nop
   1edaa: 00 00        	s.nop16
   1edac: 00 00        	s.nop16
   1edae: 1e 93        	<unknown>
   1edb0: 00 00 00 00  	v.nop
   1edb4: 00 00 00 00 00 00    	m.nop
   1edba: 31 17 c6 38  	l.st.w r6, [sp, 0x39c]
   1edbe: 01 00        	c.nop
   1edc0: 00 00 00 00  	v.nop
   1edc4: 00 00 00 00 00 00    	m.nop
   1edca: 2b 60 82 01  	l.jal r0, 0x2c <0x1f080>
   1edce: 01 00        	c.nop
   1edd0: 00 00 00 00  	v.nop
   1edd4: 00 00 00 00 00 00    	m.nop
   1edda: 73 b9 83 00  	l.movs r5, md23_wrk
   1edde: 11 43        	<unknown>
   1ede0: 00 00 00 00  	v.nop
   1ede4: 00 00 00 00 00 00    	m.nop
   1edea: 65 29 86 01  	l.andix r5, r5, 0x6
   1edee: 01 00        	c.nop
   1edf0: 00 00 00 00  	v.nop
   1edf4: 00 00 00 00 00 00    	m.nop
   1edfa: 7b 2f 40 7f  	<unknown>
   1edfe: 01 00        	c.nop
   1ee00: 00 00 00 00  	v.nop
   1ee04: 00 00 00 00 00 00    	m.nop
   1ee0a: 2b 60 fe 3f  	l.jal r0, -0x14 <0x1ecc0>
   1ee0e: 01 00        	c.nop
   1ee10: 00 00 00 00  	v.nop
   1ee14: 00 00 00 00 00 00    	m.nop
   1ee1a: 73 01 80 00  	l.movs r5, sip_id
   1ee1e: 01 00        	c.nop
   1ee20: 00 00 00 00  	v.nop
   1ee24: 00 00 00 00 00 00    	m.nop
   1ee2a: ef 37 80 20  	l.ldi_hi.m1 r6, 0x1f
   1ee2e: c2 02        	<unknown>
   1ee30: 00 00 00 00  	v.nop
   1ee34: 00 00 00 00 00 00    	m.nop
   1ee3a: 65 29 86 01  	l.andix r5, r5, 0x6
   1ee3e: 01 00        	c.nop
   1ee40: 00 00 00 00  	v.nop
   1ee44: 00 00 00 00 00 00    	m.nop
   1ee4a: 2f 30 08 15  	l.ldi16.u r6, 0xa100
   1ee4e: 01 00        	c.nop
   1ee50: 00 00 00 00  	v.nop
   1ee54: 00 00 00 00 00 00    	m.nop
   1ee5a: 2f 30 90 00  	l.ldi_hi r6, 0x200
   1ee5e: 01 00        	c.nop
   1ee60: 00 00 00 00  	v.nop
   1ee64: 00 00 00 00 00 00    	m.nop
   1ee6a: 65 29 06 02  	l.orix r5, r5, 0x6
   1ee6e: 01 00        	c.nop
   1ee70: 00 00 00 00  	v.nop
   1ee74: 00 00 00 00 00 00    	m.nop
   1ee7a: b1 11 00 3c  	l.ld.b r6, [sp, 0x3c0]
   1ee7e: 01 00        	c.nop
   1ee80: 00 00 00 00  	v.nop
   1ee84: 00 00 00 00 00 00    	m.nop
   1ee8a: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1ee8e: 01 00        	c.nop
   1ee90: 00 00 00 00  	v.nop
   1ee94: 00 00 00 00 00 00    	m.nop
   1ee9a: fb 32 60 7f  	<unknown>
   1ee9e: 01 00        	c.nop
   1eea0: 00 00 00 00  	v.nop
   1eea4: 00 00 00 00 00 00    	m.nop
   1eeaa: b1 11 00 3a  	l.ld.b r6, [sp, 0x3a0]
   1eeae: 01 00        	c.nop
   1eeb0: 00 00 00 00  	v.nop
   1eeb4: 00 00 00 00 00 00    	m.nop
   1eeba: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1eebe: 01 00        	c.nop
   1eec0: 00 00 00 00  	v.nop
   1eec4: 00 00 00 00 00 00    	m.nop
   1eeca: fb 34 20 00  	<unknown>
   1eece: 01 00        	c.nop
   1eed0: 00 00 00 00  	v.nop
   1eed4: 00 00 00 00 00 00    	m.nop
   1eeda: b1 11 09 38  	l.ld.b r6, [sp, 0x389]
   1eede: 01 00        	c.nop
   1eee0: 00 00 00 00  	v.nop
   1eee4: 00 00 00 00 00 00    	m.nop
   1eeea: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1eeee: 01 00        	c.nop
   1eef0: 00 00 00 00  	v.nop
   1eef4: 00 00 00 00 00 00    	m.nop
   1eefa: bb 30 00 40  	<unknown>
   1eefe: 01 00        	c.nop
   1ef00: 00 00 00 00  	v.nop
   1ef04: 00 00 00 00 00 00    	m.nop
   1ef0a: 2b 60 84 03  	l.jal r0, 0x84c <0x273c0>
   1ef0e: 01 00        	c.nop
   1ef10: 00 00 00 00  	v.nop
   1ef14: 00 00 00 00 00 00    	m.nop
   1ef1a: 31 12 40 08  	l.ld.w r8, [sp, 0x80]
   1ef1e: 01 00        	c.nop
   1ef20: 00 00 00 00  	v.nop
   1ef24: 00 00 00 00 00 00    	m.nop
   1ef2a: b1 41 40 00  	l.ld.w r6, [r8]
   1ef2e: 01 00        	c.nop
   1ef30: 00 00 00 00  	v.nop
   1ef34: 00 00 00 00 00 00    	m.nop
   1ef3a: f1 41 44 00  	l.ld.w r7, [r8, 0x4]
   1ef3e: 01 00        	c.nop
   1ef40: 00 00 00 00  	v.nop
   1ef44: 00 00 00 00 00 00    	m.nop
   1ef4a: 31 42 48 00  	l.ld.w r8, [r8, 0x8]
   1ef4e: 01 00        	c.nop
   1ef50: 00 00 00 00  	v.nop
   1ef54: 00 00 00 00 00 00    	m.nop
   1ef5a: a3 39 06 19  	<unknown>
   1ef5e: 01 00        	c.nop
   1ef60: 00 00 00 00  	v.nop
   1ef64: 00 00 00 00 00 00    	m.nop
   1ef6a: a3 31 08 19  	<unknown>
   1ef6e: 01 00        	c.nop
   1ef70: 00 00 00 00  	v.nop
   1ef74: 00 00 00 00 00 00    	m.nop
   1ef7a: 2b 30 84 03  	l.jal r0, 0x846 <0x273d0>
   1ef7e: 01 00        	c.nop
   1ef80: 00 00 00 00  	v.nop
   1ef84: 00 00 00 00 00 00    	m.nop
   1ef8a: 00 00        	s.nop16
   1ef8c: 00 00        	s.nop16
   1ef8e: 81 42        	<unknown>
   1ef90: 00 00 00 00  	v.nop
   1ef94: 00 00 00 00 00 00    	m.nop
   1ef9a: b1 11 00 3c  	l.ld.b r6, [sp, 0x3c0]
   1ef9e: 01 00        	c.nop
   1efa0: 00 00 00 00  	v.nop
   1efa4: 00 00 00 00 00 00    	m.nop
   1efaa: a5 31 9c 01  	l.andix r6, r6, 0x1c
   1efae: 01 00        	c.nop
   1efb0: 00 00 00 00  	v.nop
   1efb4: 00 00 00 00 00 00    	m.nop
   1efba: 7b 36 40 7f  	<unknown>
   1efbe: 01 00        	c.nop
   1efc0: 00 00 00 00  	v.nop
   1efc4: 00 00 00 00 00 00    	m.nop
   1efca: 2b 70 fe 3f  	l.jal r0, -0x12 <0x1eea0>
   1efce: 01 00        	c.nop
   1efd0: 00 00 00 00  	v.nop
   1efd4: 00 00 00 00 00 00    	m.nop
   1efda: bb 32 00 00  	<unknown>
   1efde: 01 00        	c.nop
   1efe0: 00 00 00 00  	v.nop
   1efe4: 00 00 00 00 00 00    	m.nop
   1efea: 31 12 50 36  	l.ld.w r8, [sp, 0x370]
   1efee: 01 00        	c.nop
   1eff0: 00 00 00 00  	v.nop
   1eff4: 00 00 00 00 00 00    	m.nop
   1effa: b1 11 5c 36  	l.ld.w r6, [sp, 0x37c]
   1effe: 01 00        	c.nop
   1f000: 00 00 00 00  	v.nop
   1f004: 00 00 00 00 00 00    	m.nop
   1f00a: 2f 38 08 29  	l.ldi16.s r7, 0x4100
   1f00e: 01 00        	c.nop
   1f010: 00 00 00 00  	v.nop
   1f014: 00 00 00 00 00 00    	m.nop
   1f01a: af 3f 91 08  	l.ldi_hi r7, 0x423e
   1f01e: 01 00        	c.nop
   1f020: 00 00 00 00  	v.nop
   1f024: 00 00 00 00 00 00    	m.nop
   1f02a: fb 40 0f 00  	<unknown>
   1f02e: 01 00        	c.nop
   1f030: 00 00 00 00  	v.nop
   1f034: 00 00 00 00 00 00    	m.nop
   1f03a: 2f 38 08 15  	l.ldi16.u r7, 0xa100
   1f03e: 01 00        	c.nop
   1f040: 00 00 00 00  	v.nop
   1f044: 00 00 00 00 00 00    	m.nop
   1f04a: 2f 38 80 08  	l.ldi_hi r7, 0x4000
   1f04e: 01 00        	c.nop
   1f050: 00 00 00 00  	v.nop
   1f054: 00 00 00 00 00 00    	m.nop
   1f05a: 31 38 c6 00  	l.st.w r6, [r7]
   1f05e: 01 00        	c.nop
   1f060: 00 00 00 00  	v.nop
   1f064: 00 00 00 00 00 00    	m.nop
   1f06a: 2b 30 81 01  	l.jal r0, 0x16 <0x1f1c0>
   1f06e: 01 00        	c.nop
   1f070: 00 00 00 00  	v.nop
   1f074: 00 00 00 00 00 00    	m.nop
   1f07a: b1 11 5c 38  	l.ld.w r6, [sp, 0x39c]
   1f07e: 01 00        	c.nop
   1f080: 00 00 00 00  	v.nop
   1f084: 00 00 00 00 00 00    	m.nop
   1f08a: 31 12 54 38  	l.ld.w r8, [sp, 0x394]
   1f08e: 01 00        	c.nop
   1f090: 00 00 00 00  	v.nop
   1f094: 00 00 00 00 00 00    	m.nop
   1f09a: f1 11 50 38  	l.ld.w r7, [sp, 0x390]
   1f09e: 01 00        	c.nop
   1f0a0: 00 00 00 00  	v.nop
   1f0a4: 00 00 00 00 00 00    	m.nop
   1f0aa: ef 57 7f 1f  	l.ldi16.u r10, 0xffff
   1f0ae: 01 00        	c.nop
   1f0b0: 00 00 00 00  	v.nop
   1f0b4: 00 00 00 00 00 00    	m.nop
   1f0ba: ef 57 83 00  	l.ldi_hi r10, 0x7f
   1f0be: 01 00        	c.nop
   1f0c0: 00 00 00 00  	v.nop
   1f0c4: 00 00 00 00 00 00    	m.nop
   1f0ca: 2f 50 00 25  	l.ldi16.s r10, 0x2000
   1f0ce: aa 85        	<unknown>
   1f0d0: 00 00 00 00  	v.nop
   1f0d4: 00 00 00 00 00 00    	m.nop
   1f0da: 71 12 5c 36  	l.ld.w r9, [sp, 0x37c]
   1f0de: 12 04        	<unknown>
   1f0e0: 00 00 00 00  	v.nop
   1f0e4: 00 00 00 00 00 00    	m.nop
   1f0ea: 6f 57 91 08  	l.ldi_hi r10, 0x423d
   1f0ee: e6 03        	<unknown>
   1f0f0: 00 00 00 00  	v.nop
   1f0f4: 00 00 00 00 00 00    	m.nop
   1f0fa: 2f 56 33 2b  	l.ldi16.s r10, 0x5678
   1f0fe: 2a 94        	<unknown>
   1f100: 00 00 00 00  	v.nop
   1f104: 00 00 00 00 00 00    	m.nop
   1f10a: 2f 55 91 02  	l.ldi_hi r10, 0x1234
   1f10e: 01 00        	c.nop
   1f110: 00 00 00 00  	v.nop
   1f114: 00 00 00 00 00 00    	m.nop
   1f11a: a5 31 8b 01  	l.andix r6, r6, 0xb
   1f11e: 01 00        	c.nop
   1f120: 00 00 00 00  	v.nop
   1f124: 00 00 00 00 00 00    	m.nop
   1f12a: 31 40 ca 00  	l.st.w r10, [r8]
   1f12e: 01 00        	c.nop
   1f130: 00 00 00 00  	v.nop
   1f134: 00 00 00 00 00 00    	m.nop
   1f13a: a5 3a 0b 02  	l.orix r10, r7, 0xb
   1f13e: 01 00        	c.nop
   1f140: 00 00 00 00  	v.nop
   1f144: 00 00 00 00 00 00    	m.nop
   1f14a: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   1f14e: 01 00        	c.nop
   1f150: 00 00 00 00  	v.nop
   1f154: 00 00 00 00 00 00    	m.nop
   1f15a: ef 5f ff 00  	l.ldi_hi r11, 0xfff
   1f15e: 01 00        	c.nop
   1f160: 00 00 00 00  	v.nop
   1f164: 00 00 00 00 00 00    	m.nop
   1f16a: a5 39 06 02  	l.orix r6, r7, 0x6
   1f16e: 01 00        	c.nop
   1f170: 00 00 00 00  	v.nop
   1f174: 00 00 00 00 00 00    	m.nop
   1f17a: 31 41 cb 00  	l.st.w r11, [r8, 0x4]
   1f17e: 01 00        	c.nop
   1f180: 00 00 00 00  	v.nop
   1f184: 00 00 00 00 00 00    	m.nop
   1f18a: 31 42 ca 00  	l.st.w r10, [r8, 0x8]
   1f18e: 01 00        	c.nop
   1f190: 00 00 00 00  	v.nop
   1f194: 00 00 00 00 00 00    	m.nop
   1f19a: 31 40 c9 00  	l.st.w r9, [r8]
   1f19e: 01 00        	c.nop
   1f1a0: 00 00 00 00  	v.nop
   1f1a4: 00 00 00 00 00 00    	m.nop
   1f1aa: 31 41 c5 00  	l.st.w r5, [r8, 0x4]
   1f1ae: 01 00        	c.nop
   1f1b0: 00 00 00 00  	v.nop
   1f1b4: 00 00 00 00 00 00    	m.nop
   1f1ba: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   1f1be: 01 00        	c.nop
   1f1c0: 00 00 00 00  	v.nop
   1f1c4: 00 00 00 00 00 00    	m.nop
   1f1ca: b1 11 5c 3a  	l.ld.w r6, [sp, 0x3bc]
   1f1ce: 01 00        	c.nop
   1f1d0: 00 00 00 00  	v.nop
   1f1d4: 00 00 00 00 00 00    	m.nop
   1f1da: 00 00        	s.nop16
   1f1dc: 00 00        	s.nop16
   1f1de: 05 03        	<unknown>
   1f1e0: 00 00 00 00  	v.nop
   1f1e4: 00 00 00 00 00 00    	m.nop
   1f1ea: 31 17 c6 3a  	l.st.w r6, [sp, 0x3bc]
   1f1ee: 01 00        	c.nop
   1f1f0: 00 00 00 00  	v.nop
   1f1f4: 00 00 00 00 00 00    	m.nop
   1f1fa: f1 11 0c 3c  	l.ld.b r7, [sp, 0x3cc]
   1f1fe: 01 00        	c.nop
   1f200: 00 00 00 00  	v.nop
   1f204: 00 00 00 00 00 00    	m.nop
   1f20a: a3 a1 99 18  	<unknown>
   1f20e: 01 00        	c.nop
   1f210: 00 00 00 00  	v.nop
   1f214: 00 00 00 00 00 00    	m.nop
   1f21a: e5 39 9c 01  	l.andix r7, r7, 0x1c
   1f21e: 01 00        	c.nop
   1f220: 00 00 00 00  	v.nop
   1f224: 00 00 00 00 00 00    	m.nop
   1f22a: 3b 3b 20 00  	<unknown>
   1f22e: 01 00        	c.nop
   1f230: 00 00 00 00  	v.nop
   1f234: 00 00 00 00 00 00    	m.nop
   1f23a: f1 11 48 3c  	l.ld.w r7, [sp, 0x3c8]
   1f23e: 01 00        	c.nop
   1f240: 00 00 00 00  	v.nop
   1f244: 00 00 00 00 00 00    	m.nop
   1f24a: 31 12 44 3c  	l.ld.w r8, [sp, 0x3c4]
   1f24e: 01 00        	c.nop
   1f250: 00 00 00 00  	v.nop
   1f254: 00 00 00 00 00 00    	m.nop
   1f25a: fb 39 08 40  	<unknown>
   1f25e: 01 00        	c.nop
   1f260: 00 00 00 00  	v.nop
   1f264: 00 00 00 00 00 00    	m.nop
   1f26a: f1 11 54 3a  	l.ld.w r7, [sp, 0x3b4]
   1f26e: 01 00        	c.nop
   1f270: 00 00 00 00  	v.nop
   1f274: 00 00 00 00 00 00    	m.nop
   1f27a: 31 12 50 3a  	l.ld.w r8, [sp, 0x3b0]
   1f27e: 01 00        	c.nop
   1f280: 00 00 00 00  	v.nop
   1f284: 00 00 00 00 00 00    	m.nop
   1f28a: 31 12 cf 3c  	l.st.w r15, [sp, 0x3c8]
   1f28e: 01 00        	c.nop
   1f290: 00 00 00 00  	v.nop
   1f294: 00 00 00 00 00 00    	m.nop
   1f29a: 00 00        	s.nop16
   1f29c: 00 00        	s.nop16
   1f29e: a2 93        	<unknown>
   1f2a0: 00 00 00 00  	v.nop
   1f2a4: 00 00 00 00 00 00    	m.nop
   1f2aa: 31 15 c7 3a  	l.st.w r7, [sp, 0x3b4]
   1f2ae: 01 00        	c.nop
   1f2b0: 00 00 00 00  	v.nop
   1f2b4: 00 00 00 00 00 00    	m.nop
   1f2ba: 2b 60 80 01  	l.jal r0, 0xc <0x1f370>
   1f2be: 01 00        	c.nop
   1f2c0: 00 00 00 00  	v.nop
   1f2c4: 00 00 00 00 00 00    	m.nop
   1f2ca: bb 3a 00 00  	<unknown>
   1f2ce: 01 00        	c.nop
   1f2d0: 00 00 00 00  	v.nop
   1f2d4: 00 00 00 00 00 00    	m.nop
   1f2da: 31 12 50 36  	l.ld.w r8, [sp, 0x370]
   1f2de: 01 00        	c.nop
   1f2e0: 00 00 00 00  	v.nop
   1f2e4: 00 00 00 00 00 00    	m.nop
   1f2ea: 71 11 40 38  	l.ld.w r5, [sp, 0x380]
   1f2ee: 01 00        	c.nop
   1f2f0: 00 00 00 00  	v.nop
   1f2f4: 00 00 00 00 00 00    	m.nop
   1f2fa: 2f 38 08 29  	l.ldi16.s r7, 0x4100
   1f2fe: 01 00        	c.nop
   1f300: 00 00 00 00  	v.nop
   1f304: 00 00 00 00 00 00    	m.nop
   1f30a: af 3f 91 08  	l.ldi_hi r7, 0x423e
   1f30e: 01 00        	c.nop
   1f310: 00 00 00 00  	v.nop
   1f314: 00 00 00 00 00 00    	m.nop
   1f31a: fb 40 0f 00  	<unknown>
   1f31e: 01 00        	c.nop
   1f320: 00 00 00 00  	v.nop
   1f324: 00 00 00 00 00 00    	m.nop
   1f32a: 2f 38 08 15  	l.ldi16.u r7, 0xa100
   1f32e: 01 00        	c.nop
   1f330: 00 00 00 00  	v.nop
   1f334: 00 00 00 00 00 00    	m.nop
   1f33a: 2f 38 80 08  	l.ldi_hi r7, 0x4000
   1f33e: 01 00        	c.nop
   1f340: 00 00 00 00  	v.nop
   1f344: 00 00 00 00 00 00    	m.nop
   1f34a: 31 38 c5 00  	l.st.w r5, [r7]
   1f34e: 01 00        	c.nop
   1f350: 00 00 00 00  	v.nop
   1f354: 00 00 00 00 00 00    	m.nop
   1f35a: 2b 30 81 01  	l.jal r0, 0x16 <0x1f4b0>
   1f35e: 01 00        	c.nop
   1f360: 00 00 00 00  	v.nop
   1f364: 00 00 00 00 00 00    	m.nop
   1f36a: f1 11 54 3a  	l.ld.w r7, [sp, 0x3b4]
   1f36e: 01 00        	c.nop
   1f370: 00 00 00 00  	v.nop
   1f374: 00 00 00 00 00 00    	m.nop
   1f37a: 71 12 4c 3a  	l.ld.w r9, [sp, 0x3ac]
   1f37e: 01 00        	c.nop
   1f380: 00 00 00 00  	v.nop
   1f384: 00 00 00 00 00 00    	m.nop
   1f38a: 31 12 48 3a  	l.ld.w r8, [sp, 0x3a8]
   1f38e: 01 00        	c.nop
   1f390: 00 00 00 00  	v.nop
   1f394: 00 00 00 00 00 00    	m.nop
   1f39a: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   1f39e: 01 00        	c.nop
   1f3a0: 00 00 00 00  	v.nop
   1f3a4: 00 00 00 00 00 00    	m.nop
   1f3aa: ef 5f 83 00  	l.ldi_hi r11, 0x7f
   1f3ae: 01 00        	c.nop
   1f3b0: 00 00 00 00  	v.nop
   1f3b4: 00 00 00 00 00 00    	m.nop
   1f3ba: 2f 58 00 25  	l.ldi16.s r11, 0x2000
   1f3be: 2e 86        	<unknown>
   1f3c0: 00 00 00 00  	v.nop
   1f3c4: 00 00 00 00 00 00    	m.nop
   1f3ca: b1 12 40 38  	l.ld.w r10, [sp, 0x380]
   1f3ce: 92 04        	<unknown>
   1f3d0: 00 00 00 00  	v.nop
   1f3d4: 00 00 00 00 00 00    	m.nop
   1f3da: 6f 5f 91 08  	l.ldi_hi r11, 0x423d
   1f3de: 66 04        	<unknown>
   1f3e0: 00 00 00 00  	v.nop
   1f3e4: 00 00 00 00 00 00    	m.nop
   1f3ea: 2f 5e 33 2b  	l.ldi16.s r11, 0x5678
   1f3ee: ae 94        	<unknown>
   1f3f0: 00 00 00 00  	v.nop
   1f3f4: 00 00 00 00 00 00    	m.nop
   1f3fa: 2f 5d 91 02  	l.ldi_hi r11, 0x1234
   1f3fe: 01 00        	c.nop
   1f400: 00 00 00 00  	v.nop
   1f404: 00 00 00 00 00 00    	m.nop
   1f40a: e5 39 8c 01  	l.andix r7, r7, 0xc
   1f40e: 01 00        	c.nop
   1f410: 00 00 00 00  	v.nop
   1f414: 00 00 00 00 00 00    	m.nop
   1f41a: 31 48 cb 00  	l.st.w r11, [r9]
   1f41e: 01 00        	c.nop
   1f420: 00 00 00 00  	v.nop
   1f424: 00 00 00 00 00 00    	m.nop
   1f42a: e5 42 0c 02  	l.orix r11, r8, 0xc
   1f42e: 01 00        	c.nop
   1f430: 00 00 00 00  	v.nop
   1f434: 00 00 00 00 00 00    	m.nop
   1f43a: ef 67 7f 1f  	l.ldi16.u r12, 0xffff
   1f43e: 01 00        	c.nop
   1f440: 00 00 00 00  	v.nop
   1f444: 00 00 00 00 00 00    	m.nop
   1f44a: ef 67 ff 00  	l.ldi_hi r12, 0xfff
   1f44e: 01 00        	c.nop
   1f450: 00 00 00 00  	v.nop
   1f454: 00 00 00 00 00 00    	m.nop
   1f45a: e5 41 07 02  	l.orix r7, r8, 0x7
   1f45e: 01 00        	c.nop
   1f460: 00 00 00 00  	v.nop
   1f464: 00 00 00 00 00 00    	m.nop
   1f46a: 31 49 cc 00  	l.st.w r12, [r9, 0x4]
   1f46e: 01 00        	c.nop
   1f470: 00 00 00 00  	v.nop
   1f474: 00 00 00 00 00 00    	m.nop
   1f47a: 31 4a cb 00  	l.st.w r11, [r9, 0x8]
   1f47e: 01 00        	c.nop
   1f480: 00 00 00 00  	v.nop
   1f484: 00 00 00 00 00 00    	m.nop
   1f48a: 31 48 ca 00  	l.st.w r10, [r9]
   1f48e: 01 00        	c.nop
   1f490: 00 00 00 00  	v.nop
   1f494: 00 00 00 00 00 00    	m.nop
   1f49a: 31 49 c5 00  	l.st.w r5, [r9, 0x4]
   1f49e: 01 00        	c.nop
   1f4a0: 00 00 00 00  	v.nop
   1f4a4: 00 00 00 00 00 00    	m.nop
   1f4aa: 31 4a c7 00  	l.st.w r7, [r9, 0x8]
   1f4ae: 01 00        	c.nop
   1f4b0: 00 00 00 00  	v.nop
   1f4b4: 00 00 00 00 00 00    	m.nop
   1f4ba: 71 11 48 3c  	l.ld.w r5, [sp, 0x3c8]
   1f4be: 01 00        	c.nop
   1f4c0: 00 00 00 00  	v.nop
   1f4c4: 00 00 00 00 00 00    	m.nop
   1f4ca: 00 00        	s.nop16
   1f4cc: 00 00        	s.nop16
   1f4ce: 85 02        	<unknown>
   1f4d0: 00 00 00 00  	v.nop
   1f4d4: 00 00 00 00 00 00    	m.nop
   1f4da: 31 12 c5 3c  	l.st.w r5, [sp, 0x3c8]
   1f4de: 01 00        	c.nop
   1f4e0: 00 00 00 00  	v.nop
   1f4e4: 00 00 00 00 00 00    	m.nop
   1f4ea: 2f 2d 01 21  	l.ldi16.s r5, 0x34
   1f4ee: 01 00        	c.nop
   1f4f0: 00 00 00 00  	v.nop
   1f4f4: 00 00 00 00 00 00    	m.nop
   1f4fa: 00 00        	s.nop16
   1f4fc: 00 00        	s.nop16
   1f4fe: 16 92        	<unknown>
   1f500: 00 00 00 00  	v.nop
   1f504: 00 00 00 00 00 00    	m.nop
   1f50a: 31 20 da 00  	l.st.w r26, [r4]
   1f50e: 01 00        	c.nop
   1f510: 00 00 00 00  	v.nop
   1f514: 00 00 00 00 00 00    	m.nop
   1f51a: 2f 20 02 21  	l.ldi16.s r4, 0x40
   1f51e: 01 00        	c.nop
   1f520: 00 00 00 00  	v.nop
   1f524: 00 00 00 00 00 00    	m.nop
   1f52a: 69 21 86 18  	<unknown>
   1f52e: 01 00        	c.nop
   1f530: 00 00 00 00  	v.nop
   1f534: 00 00 00 00 00 00    	m.nop
   1f53a: 31 11 58 12  	l.ld.w r4, [sp, 0x138]
   1f53e: 01 00        	c.nop
   1f540: 00 00 00 00  	v.nop
   1f544: 00 00 00 00 00 00    	m.nop
   1f54a: 31 11 cf 34  	l.st.w r15, [sp, 0x344]
   1f54e: 01 00        	c.nop
   1f550: 00 00 00 00  	v.nop
   1f554: 00 00 00 00 00 00    	m.nop
   1f55a: 31 16 c5 1e  	l.st.w r5, [sp, 0x1f8]
   1f55e: 01 00        	c.nop
   1f560: 00 00 00 00  	v.nop
   1f564: 00 00 00 00 00 00    	m.nop
   1f56a: 31 10 c4 34  	l.st.w r4, [sp, 0x340]
   1f56e: 01 00        	c.nop
   1f570: 00 00 00 00  	v.nop
   1f574: 00 00 00 00 00 00    	m.nop
   1f57a: bb a8 9a 80  	l.bge.s r21, r26, 0x2 <0x1f590>
   1f57e: 01 00        	c.nop
   1f580: 00 00 00 00  	v.nop
   1f584: 00 00 00 00 00 00    	m.nop
   1f58a: 2b 18 ed 01  	l.jal r0, 0x6d3 <0x262b0>
   1f58e: 01 00        	c.nop
   1f590: 00 00 00 00  	v.nop
   1f594: 00 00 00 00 00 00    	m.nop
   1f59a: 31 11 58 08  	l.ld.w r4, [sp, 0x98]
   1f59e: 01 4a        	<unknown>
   1f5a0: 00 00 00 00  	v.nop
   1f5a4: 00 00 00 00 00 00    	m.nop
   1f5aa: b1 11 44 08  	l.ld.w r6, [sp, 0x84]
   1f5ae: 52 8f        	<unknown>
   1f5b0: 00 00 00 00  	v.nop
   1f5b4: 00 00 00 00 00 00    	m.nop
   1f5ba: 31 16 d5 06  	l.st.w r21, [sp, 0x78]
   1f5be: 01 00        	c.nop
   1f5c0: 00 00 00 00  	v.nop
   1f5c4: 00 00 00 00 00 00    	m.nop
   1f5ca: 23 c9 04 18  	<unknown>
   1f5ce: 01 00        	c.nop
   1f5d0: 00 00 00 00  	v.nop
   1f5d4: 00 00 00 00 00 00    	m.nop
   1f5da: 31 11 c6 04  	l.st.w r6, [sp, 0x44]
   1f5de: 01 00        	c.nop
   1f5e0: 00 00 00 00  	v.nop
   1f5e4: 00 00 00 00 00 00    	m.nop
   1f5ea: a1 21 01 00  	l.addi.u r6, r4, 0x1
   1f5ee: 01 00        	c.nop
   1f5f0: 00 00 00 00  	v.nop
   1f5f4: 00 00 00 00 00 00    	m.nop
   1f5fa: 31 15 c6 20  	l.st.w r6, [sp, 0x214]
   1f5fe: 01 00        	c.nop
   1f600: 00 00 00 00  	v.nop
   1f604: 00 00 00 00 00 00    	m.nop
   1f60a: b1 11 4c 1a  	l.ld.w r6, [sp, 0x1ac]
   1f60e: 01 00        	c.nop
   1f610: 00 00 00 00  	v.nop
   1f614: 00 00 00 00 00 00    	m.nop
   1f61a: 31 16 c4 20  	l.st.w r4, [sp, 0x218]
   1f61e: 16 92        	<unknown>
   1f620: 00 00 00 00  	v.nop
   1f624: 00 00 00 00 00 00    	m.nop
   1f62a: 31 10 c4 16  	l.st.w r4, [sp, 0x160]
   1f62e: 01 00        	c.nop
   1f630: 00 00 00 00  	v.nop
   1f634: 00 00 00 00 00 00    	m.nop
   1f63a: 23 29 08 03  	l.slli r4, r5, 0x8
   1f63e: 01 00        	c.nop
   1f640: 00 00 00 00  	v.nop
   1f644: 00 00 00 00 00 00    	m.nop
   1f64a: 31 16 c4 0c  	l.st.w r4, [sp, 0xd8]
   1f64e: 01 00        	c.nop
   1f650: 00 00 00 00  	v.nop
   1f654: 00 00 00 00 00 00    	m.nop
   1f65a: 31 10 c6 06  	l.st.w r6, [sp, 0x60]
   1f65e: 01 00        	c.nop
   1f660: 00 00 00 00  	v.nop
   1f664: 00 00 00 00 00 00    	m.nop
   1f66a: 2b 68 81 01  	l.jal r0, 0x1d <0x1f830>
   1f66e: 01 00        	c.nop
   1f670: 00 00 00 00  	v.nop
   1f674: 00 00 00 00 00 00    	m.nop
   1f67a: 31 11 58 0a  	l.ld.w r4, [sp, 0xb8]
   1f67e: 01 00        	c.nop
   1f680: 00 00 00 00  	v.nop
   1f684: 00 00 00 00 00 00    	m.nop
   1f68a: 2f 28 10 11  	l.ldi16.u r5, 0x8200
   1f68e: 01 00        	c.nop
   1f690: 00 00 00 00  	v.nop
   1f694: 00 00 00 00 00 00    	m.nop
   1f69a: 6f 2d 80 00  	l.ldi_hi r5, 0x15
   1f69e: 01 00        	c.nop
   1f6a0: 00 00 00 00  	v.nop
   1f6a4: 00 00 00 00 00 00    	m.nop
   1f6aa: 23 23 05 18  	<unknown>
   1f6ae: 01 00        	c.nop
   1f6b0: 00 00 00 00  	v.nop
   1f6b4: 00 00 00 00 00 00    	m.nop
   1f6ba: 2f 20 08 21  	l.ldi16.s r4, 0x100
   1f6be: a6 82        	<unknown>
   1f6c0: 00 00 00 00  	v.nop
   1f6c4: 00 00 00 00 00 00    	m.nop
   1f6ca: 31 14 c4 00  	l.st.w r4, [sp, 0x10]
   1f6ce: 66 82        	<unknown>
   1f6d0: 00 00 00 00  	v.nop
   1f6d4: 00 00 00 00 00 00    	m.nop
   1f6da: 31 13 c9 00  	l.st.w r9, [sp, 0xc]
   1f6de: 52 83        	<unknown>
   1f6e0: 00 00 00 00  	v.nop
   1f6e4: 00 00 00 00 00 00    	m.nop
   1f6ea: 31 12 d3 00  	l.st.w r19, [sp, 0x8]
   1f6ee: a6 83        	<unknown>
   1f6f0: 00 00 00 00  	v.nop
   1f6f4: 00 00 00 00 00 00    	m.nop
   1f6fa: 31 11 c9 00  	l.st.w r9, [sp, 0x4]
   1f6fe: 5a 84        	<unknown>
   1f700: 00 00 00 00  	v.nop
   1f704: 00 00 00 00 00 00    	m.nop
   1f70a: 31 10 cc 00  	l.st.w r12, [sp]
   1f70e: 5e 85        	<unknown>
   1f710: 00 00 00 00  	v.nop
   1f714: 00 00 00 00 00 00    	m.nop
   1f71a: 31 13 50 20  	l.ld.w r12, [sp, 0x210]
   1f71e: a6 85        	<unknown>
   1f720: 00 00 00 00  	v.nop
   1f724: 00 00 00 00 00 00    	m.nop
   1f72a: eb 60 00 01  	l.jalr lr, r12, 0x0
   1f72e: 01 00        	c.nop
   1f730: 00 00 00 00  	v.nop
   1f734: 00 00 00 00 00 00    	m.nop
   1f73a: 31 11 48 04  	l.ld.w r4, [sp, 0x48]
   1f73e: 46 8a        	<unknown>
   1f740: 00 00 00 00  	v.nop
   1f744: 00 00 00 00 00 00    	m.nop
   1f74a: 71 11 5c 22  	l.ld.w r5, [sp, 0x23c]
   1f74e: 81 47        	<unknown>
   1f750: 00 00 00 00  	v.nop
   1f754: 00 00 00 00 00 00    	m.nop
   1f75a: 71 16 50 06  	l.ld.w r25, [sp, 0x70]
   1f75e: 09 4b        	<unknown>
   1f760: 00 00 00 00  	v.nop
   1f764: 00 00 00 00 00 00    	m.nop
   1f76a: 23 d5 94 18  	<unknown>
   1f76e: 01 00        	c.nop
   1f770: 00 00 00 00  	v.nop
   1f774: 00 00 00 00 00 00    	m.nop
   1f77a: 23 d1 84 18  	<unknown>
   1f77e: 01 00        	c.nop
   1f780: 00 00 00 00  	v.nop
   1f784: 00 00 00 00 00 00    	m.nop
   1f78a: 31 12 c4 04  	l.st.w r4, [sp, 0x48]
   1f78e: 16 8f        	<unknown>
   1f790: 00 00 00 00  	v.nop
   1f794: 00 00 00 00 00 00    	m.nop
   1f79a: 31 11 40 06  	l.ld.w r4, [sp, 0x60]
   1f79e: 01 00        	c.nop
   1f7a0: 00 00 00 00  	v.nop
   1f7a4: 00 00 00 00 00 00    	m.nop
   1f7aa: 21 21 00 2f  	l.addi.s r4, r4, -0x100
   1f7ae: 01 00        	c.nop
   1f7b0: 00 00 00 00  	v.nop
   1f7b4: 00 00 00 00 00 00    	m.nop
   1f7ba: 31 10 c4 06  	l.st.w r4, [sp, 0x60]
   1f7be: 01 00        	c.nop
   1f7c0: 00 00 00 00  	v.nop
   1f7c4: 00 00 00 00 00 00    	m.nop
   1f7ca: 31 11 44 04  	l.ld.w r4, [sp, 0x44]
   1f7ce: 01 00        	c.nop
   1f7d0: 00 00 00 00  	v.nop
   1f7d4: 00 00 00 00 00 00    	m.nop
   1f7da: 21 21 00 01  	l.addi.u r4, r4, 0x100
   1f7de: 01 00        	c.nop
   1f7e0: 00 00 00 00  	v.nop
   1f7e4: 00 00 00 00 00 00    	m.nop
   1f7ea: 31 11 c4 04  	l.st.w r4, [sp, 0x44]
   1f7ee: 01 00        	c.nop
   1f7f0: 00 00 00 00  	v.nop
   1f7f4: 00 00 00 00 00 00    	m.nop
   1f7fa: 31 11 58 06  	l.ld.w r4, [sp, 0x78]
   1f7fe: 01 00        	c.nop
   1f800: 00 00 00 00  	v.nop
   1f804: 00 00 00 00 00 00    	m.nop
   1f80a: 21 21 00 2f  	l.addi.s r4, r4, -0x100
   1f80e: 01 00        	c.nop
   1f810: 00 00 00 00  	v.nop
   1f814: 00 00 00 00 00 00    	m.nop
   1f81a: 31 16 c4 06  	l.st.w r4, [sp, 0x78]
   1f81e: 01 00        	c.nop
   1f820: 00 00 00 00  	v.nop
   1f824: 00 00 00 00 00 00    	m.nop
   1f82a: 7b 2a b5 8d  	l.bge.s r5, r21, 0x6a9 <0x262b0>
   1f82e: 01 00        	c.nop
   1f830: 00 00 00 00  	v.nop
   1f834: 00 00 00 00 00 00    	m.nop
   1f83a: 3b d9 00 00  	<unknown>
   1f83e: 01 00        	c.nop
   1f840: 00 00 00 00  	v.nop
   1f844: 00 00 00 00 00 00    	m.nop
   1f84a: 31 11 40 16  	l.ld.w r4, [sp, 0x160]
   1f84e: 01 00        	c.nop
   1f850: 00 00 00 00  	v.nop
   1f854: 00 00 00 00 00 00    	m.nop
   1f85a: bb f0 84 c0  	l.blt.s r30, r4, 0x2 <0x1f870>
   1f85e: 01 00        	c.nop
   1f860: 00 00 00 00  	v.nop
   1f864: 00 00 00 00 00 00    	m.nop
   1f86a: 7b f1 20 4d  	<unknown>
   1f86e: 01 00        	c.nop
   1f870: 00 00 00 00  	v.nop
   1f874: 00 00 00 00 00 00    	m.nop
   1f87a: 31 11 40 16  	l.ld.w r4, [sp, 0x160]
   1f87e: 01 00        	c.nop
   1f880: 00 00 00 00  	v.nop
   1f884: 00 00 00 00 00 00    	m.nop
   1f88a: 61 f1 00 01  	l.addi.u r5, r30, 0x100
   1f88e: 01 00        	c.nop
   1f890: 00 00 00 00  	v.nop
   1f894: 00 00 00 00 00 00    	m.nop
   1f89a: fb 28 84 c0  	l.blt.s r5, r4, 0x3 <0x1f8c0>
   1f89e: 01 00        	c.nop
   1f8a0: 00 00 00 00  	v.nop
   1f8a4: 00 00 00 00 00 00    	m.nop
   1f8aa: 00 00        	s.nop16
   1f8ac: 00 00        	s.nop16
   1f8ae: 01 42        	<unknown>
   1f8b0: 00 00 00 00  	v.nop
   1f8b4: 00 00 00 00 00 00    	m.nop
   1f8ba: 2b 10 80 01  	l.jal r0, 0x2 <0x1f8d0>
   1f8be: 01 00        	c.nop
   1f8c0: 00 00 00 00  	v.nop
   1f8c4: 00 00 00 00 00 00    	m.nop
   1f8ca: 00 00        	s.nop16
   1f8cc: 00 00        	s.nop16
   1f8ce: 05 42        	<unknown>
   1f8d0: 00 00 00 00  	v.nop
   1f8d4: 00 00 00 00 00 00    	m.nop
   1f8da: 61 14 50 23  	l.addi.s r17, sp, 0x350
   1f8de: 01 00        	c.nop
   1f8e0: 00 00 00 00  	v.nop
   1f8e4: 00 00 00 00 00 00    	m.nop
   1f8ea: a1 11 40 23  	l.addi.s r6, sp, 0x340
   1f8ee: 01 00        	c.nop
   1f8f0: 00 00 00 00  	v.nop
   1f8f4: 00 00 00 00 00 00    	m.nop
   1f8fa: 31 15 c6 06  	l.st.w r6, [sp, 0x74]
   1f8fe: 01 00        	c.nop
   1f900: 00 00 00 00  	v.nop
   1f904: 00 00 00 00 00 00    	m.nop
   1f90a: 31 17 c5 22  	l.st.w r5, [sp, 0x23c]
   1f90e: 01 00        	c.nop
   1f910: 00 00 00 00  	v.nop
   1f914: 00 00 00 00 00 00    	m.nop
   1f91a: fb 29 d5 87  	l.bge.s r5, r21, 0x3c7 <0x23580>
   1f91e: 01 00        	c.nop
   1f920: 00 00 00 00  	v.nop
   1f924: 00 00 00 00 00 00    	m.nop
   1f92a: 71 11 40 14  	l.ld.w r5, [sp, 0x140]
   1f92e: 01 00        	c.nop
   1f930: 00 00 00 00  	v.nop
   1f934: 00 00 00 00 00 00    	m.nop
   1f93a: 25 21 05 02  	l.orix r4, r4, 0x5
   1f93e: 01 00        	c.nop
   1f940: 00 00 00 00  	v.nop
   1f944: 00 00 00 00 00 00    	m.nop
   1f94a: 3b 21 40 07  	<unknown>
   1f94e: 01 00        	c.nop
   1f950: 00 00 00 00  	v.nop
   1f954: 00 00 00 00 00 00    	m.nop
   1f95a: bb a1 1a 00  	<unknown>
   1f95e: 01 00        	c.nop
   1f960: 00 00 00 00  	v.nop
   1f964: 00 00 00 00 00 00    	m.nop
   1f96a: e1 14 d0 22  	l.addi.s r19, sp, 0x2d0
   1f96e: 01 00        	c.nop
   1f970: 00 00 00 00  	v.nop
   1f974: 00 00 00 00 00 00    	m.nop
   1f97a: a3 d2 94 18  	<unknown>
   1f97e: 01 00        	c.nop
   1f980: 00 00 00 00  	v.nop
   1f984: 00 00 00 00 00 00    	m.nop
   1f98a: bb a1 1a 40  	<unknown>
   1f98e: 01 00        	c.nop
   1f990: 00 00 00 00  	v.nop
   1f994: 00 00 00 00 00 00    	m.nop
   1f99a: 61 12 98 22  	l.addi.s r9, sp, 0x298
   1f99e: 01 00        	c.nop
   1f9a0: 00 00 00 00  	v.nop
   1f9a4: 00 00 00 00 00 00    	m.nop
   1f9aa: 2b 28 80 01  	l.jal r0, 0x5 <0x1f9f0>
   1f9ae: 01 00        	c.nop
   1f9b0: 00 00 00 00  	v.nop
   1f9b4: 00 00 00 00 00 00    	m.nop
   1f9ba: e1 14 08 23  	l.addi.s r19, sp, 0x308
   1f9be: 01 00        	c.nop
   1f9c0: 00 00 00 00  	v.nop
   1f9c4: 00 00 00 00 00 00    	m.nop
   1f9ca: a3 d2 94 18  	<unknown>
   1f9ce: 01 00        	c.nop
   1f9d0: 00 00 00 00  	v.nop
   1f9d4: 00 00 00 00 00 00    	m.nop
   1f9da: 3b a7 7a 3f  	<unknown>
   1f9de: 01 00        	c.nop
   1f9e0: 00 00 00 00  	v.nop
   1f9e4: 00 00 00 00 00 00    	m.nop
   1f9ea: 61 12 60 22  	l.addi.s r9, sp, 0x260
   1f9ee: 01 00        	c.nop
   1f9f0: 00 00 00 00  	v.nop
   1f9f4: 00 00 00 00 00 00    	m.nop
   1f9fa: 2f 24 14 21  	l.ldi16.s r4, 0x290
   1f9fe: 01 00        	c.nop
   1fa00: 00 00 00 00  	v.nop
   1fa04: 00 00 00 00 00 00    	m.nop
   1fa0a: e3 52 04 19  	<unknown>
   1fa0e: 01 00        	c.nop
   1fa10: 00 00 00 00  	v.nop
   1fa14: 00 00 00 00 00 00    	m.nop
   1fa1a: 31 11 5c 12  	l.ld.w r4, [sp, 0x13c]
   1fa1e: 01 00        	c.nop
   1fa20: 00 00 00 00  	v.nop
   1fa24: 00 00 00 00 00 00    	m.nop
   1fa2a: 31 9b 44 00  	l.ld.w r12, [r19, 0x4]
   1fa2e: 01 00        	c.nop
   1fa30: 00 00 00 00  	v.nop
   1fa34: 00 00 00 00 00 00    	m.nop
   1fa3a: b1 9b 40 00  	l.ld.w r14, [r19]
   1fa3e: 01 00        	c.nop
   1fa40: 00 00 00 00  	v.nop
   1fa44: 00 00 00 00 00 00    	m.nop
   1fa4a: a3 21 0b 18  	<unknown>
   1fa4e: 01 00        	c.nop
   1fa50: 00 00 00 00  	v.nop
   1fa54: 00 00 00 00 00 00    	m.nop
   1fa5a: 71 31 08 00  	l.ld.b r5, [r6, 0x8]
   1fa5e: 01 00        	c.nop
   1fa60: 00 00 00 00  	v.nop
   1fa64: 00 00 00 00 00 00    	m.nop
   1fa6a: 31 99 4c 02  	l.ld.w r4, [r19, 0x2c]
   1fa6e: 01 00        	c.nop
   1fa70: 00 00 00 00  	v.nop
   1fa74: 00 00 00 00 00 00    	m.nop
   1fa7a: 65 29 9c 01  	l.andix r5, r5, 0x1c
   1fa7e: 01 00        	c.nop
   1fa80: 00 00 00 00  	v.nop
   1fa84: 00 00 00 00 00 00    	m.nop
   1fa8a: 7b 29 00 00  	<unknown>
   1fa8e: 01 00        	c.nop
   1fa90: 00 00 00 00  	v.nop
   1fa94: 00 00 00 00 00 00    	m.nop
   1fa9a: bb 21 16 c0  	l.blt.u r4, r22, 0x6 <0x1faf0>
   1fa9e: 01 00        	c.nop
   1faa0: 00 00 00 00  	v.nop
   1faa4: 00 00 00 00 00 00    	m.nop
   1faaa: fb 21 16 40  	<unknown>
   1faae: 01 00        	c.nop
   1fab0: 00 00 00 00  	v.nop
   1fab4: 00 00 00 00 00 00    	m.nop
   1faba: 00 00        	s.nop16
   1fabc: 00 00        	s.nop16
   1fabe: 85 42        	<unknown>
   1fac0: 00 00 00 00  	v.nop
   1fac4: 00 00 00 00 00 00    	m.nop
   1faca: 2b 30 80 01  	l.jal r0, 0x6 <0x1fb20>
   1face: 01 00        	c.nop
   1fad0: 00 00 00 00  	v.nop
   1fad4: 00 00 00 00 00 00    	m.nop
   1fada: 00 00        	s.nop16
   1fadc: 00 00        	s.nop16
   1fade: 85 42        	<unknown>
   1fae0: 00 00 00 00  	v.nop
   1fae4: 00 00 00 00 00 00    	m.nop
   1faea: 3b 21 05 00  	<unknown>
   1faee: 01 00        	c.nop
   1faf0: 00 00 00 00  	v.nop
   1faf4: 00 00 00 00 00 00    	m.nop
   1fafa: 00 00        	s.nop16
   1fafc: 00 00        	s.nop16
   1fafe: 81 42        	<unknown>
   1fb00: 00 00 00 00  	v.nop
   1fb04: 00 00 00 00 00 00    	m.nop
   1fb0a: 2b 10 80 01  	l.jal r0, 0x2 <0x1fb20>
   1fb0e: 01 00        	c.nop
   1fb10: 00 00 00 00  	v.nop
   1fb14: 00 00 00 00 00 00    	m.nop
   1fb1a: 00 00        	s.nop16
   1fb1c: 00 00        	s.nop16
   1fb1e: 8d 42        	<unknown>
   1fb20: 00 00 00 00  	v.nop
   1fb24: 00 00 00 00 00 00    	m.nop
   1fb2a: 31 11 44 04  	l.ld.w r4, [sp, 0x44]
   1fb2e: 51 44        	<unknown>
   1fb30: 00 00 00 00  	v.nop
   1fb34: 00 00 00 00 00 00    	m.nop
   1fb3a: 31 37 c8 0c  	l.st.w r8, [r6, 0xdc]
   1fb3e: 01 00        	c.nop
   1fb40: 00 00 00 00  	v.nop
   1fb44: 00 00 00 00 00 00    	m.nop
   1fb4a: 31 12 54 04  	l.ld.w r8, [sp, 0x54]
   1fb4e: 01 00        	c.nop
   1fb50: 00 00 00 00  	v.nop
   1fb54: 00 00 00 00 00 00    	m.nop
   1fb5a: f1 99 48 00  	l.ld.w r7, [r19, 0x8]
   1fb5e: 01 00        	c.nop
   1fb60: 00 00 00 00  	v.nop
   1fb64: 00 00 00 00 00 00    	m.nop
   1fb6a: 71 9b 48 02  	l.ld.w r13, [r19, 0x28]
   1fb6e: 01 00        	c.nop
   1fb70: 00 00 00 00  	v.nop
   1fb74: 00 00 00 00 00 00    	m.nop
   1fb7a: 31 31 c5 0e  	l.st.w r5, [r6, 0xe4]
   1fb7e: 95 42        	<unknown>
   1fb80: 00 00 00 00  	v.nop
   1fb84: 00 00 00 00 00 00    	m.nop
   1fb8a: 21 21 00 01  	l.addi.u r4, r4, 0x100
   1fb8e: 01 00        	c.nop
   1fb90: 00 00 00 00  	v.nop
   1fb94: 00 00 00 00 00 00    	m.nop
   1fb9a: 31 37 c8 0e  	l.st.w r8, [r6, 0xfc]
   1fb9e: 01 00        	c.nop
   1fba0: 00 00 00 00  	v.nop
   1fba4: 00 00 00 00 00 00    	m.nop
   1fbaa: 31 12 5c 04  	l.ld.w r8, [sp, 0x5c]
   1fbae: 01 00        	c.nop
   1fbb0: 00 00 00 00  	v.nop
   1fbb4: 00 00 00 00 00 00    	m.nop
   1fbba: 63 29 87 18  	<unknown>
   1fbbe: 01 00        	c.nop
   1fbc0: 00 00 00 00  	v.nop
   1fbc4: 00 00 00 00 00 00    	m.nop
   1fbca: 31 35 cc 0e  	l.st.w r12, [r6, 0xf4]
   1fbce: 01 00        	c.nop
   1fbd0: 00 00 00 00  	v.nop
   1fbd4: 00 00 00 00 00 00    	m.nop
   1fbda: 31 34 ce 0e  	l.st.w r14, [r6, 0xf0]
   1fbde: 01 00        	c.nop
   1fbe0: 00 00 00 00  	v.nop
   1fbe4: 00 00 00 00 00 00    	m.nop
   1fbea: 31 37 cf 1c  	l.st.w r15, [r6, 0x1dc]
   1fbee: 01 00        	c.nop
   1fbf0: 00 00 00 00  	v.nop
   1fbf4: 00 00 00 00 00 00    	m.nop
   1fbfa: 31 36 c8 0e  	l.st.w r8, [r6, 0xf8]
   1fbfe: 01 00        	c.nop
   1fc00: 00 00 00 00  	v.nop
   1fc04: 00 00 00 00 00 00    	m.nop
   1fc0a: 31 30 cd 0e  	l.st.w r13, [r6, 0xe0]
   1fc0e: 01 00        	c.nop
   1fc10: 00 00 00 00  	v.nop
   1fc14: 00 00 00 00 00 00    	m.nop
   1fc1a: 31 30 c7 12  	l.st.w r7, [r6, 0x120]
   1fc1e: 01 00        	c.nop
   1fc20: 00 00 00 00  	v.nop
   1fc24: 00 00 00 00 00 00    	m.nop
   1fc2a: 31 13 d4 14  	l.st.w r20, [sp, 0x14c]
   1fc2e: 01 00        	c.nop
   1fc30: 00 00 00 00  	v.nop
   1fc34: 00 00 00 00 00 00    	m.nop
   1fc3a: 31 12 d1 14  	l.st.w r17, [sp, 0x148]
   1fc3e: 01 00        	c.nop
   1fc40: 00 00 00 00  	v.nop
   1fc44: 00 00 00 00 00 00    	m.nop
   1fc4a: 31 11 ca 14  	l.st.w r10, [sp, 0x144]
   1fc4e: 01 00        	c.nop
   1fc50: 00 00 00 00  	v.nop
   1fc54: 00 00 00 00 00 00    	m.nop
   1fc5a: 31 12 cb 0c  	l.st.w r11, [sp, 0xc8]
   1fc5e: 01 00        	c.nop
   1fc60: 00 00 00 00  	v.nop
   1fc64: 00 00 00 00 00 00    	m.nop
   1fc6a: 31 11 cd 0c  	l.st.w r13, [sp, 0xc4]
   1fc6e: 01 00        	c.nop
   1fc70: 00 00 00 00  	v.nop
   1fc74: 00 00 00 00 00 00    	m.nop
   1fc7a: 31 10 ce 0c  	l.st.w r14, [sp, 0xc0]
   1fc7e: 01 00        	c.nop
   1fc80: 00 00 00 00  	v.nop
   1fc84: 00 00 00 00 00 00    	m.nop
   1fc8a: bb d3 a5 80  	l.bge.s r26, r5, 0x2e <0x1ff60>
   1fc8e: 01 00        	c.nop
   1fc90: 00 00 00 00  	v.nop
   1fc94: 00 00 00 00 00 00    	m.nop
   1fc9a: 7b 2b 56 00  	<unknown>
   1fc9e: 01 00        	c.nop
   1fca0: 00 00 00 00  	v.nop
   1fca4: 00 00 00 00 00 00    	m.nop
   1fcaa: 00 00        	s.nop16
   1fcac: 00 00        	s.nop16
   1fcae: 8d 43        	<unknown>
   1fcb0: 00 00 00 00  	v.nop
   1fcb4: 00 00 00 00 00 00    	m.nop
   1fcba: bb 2a 67 00  	<unknown>
   1fcbe: 01 00        	c.nop
   1fcc0: 00 00 00 00  	v.nop
   1fcc4: 00 00 00 00 00 00    	m.nop
   1fcca: 00 00        	s.nop16
   1fccc: 00 00        	s.nop16
   1fcce: 91 43        	<unknown>
   1fcd0: 00 00 00 00  	v.nop
   1fcd4: 00 00 00 00 00 00    	m.nop
   1fcda: fb 2a 27 41  	<unknown>
   1fcde: 01 00        	c.nop
   1fce0: 00 00 00 00  	v.nop
   1fce4: 00 00 00 00 00 00    	m.nop
   1fcea: 71 11 40 02  	l.ld.w r5, [sp, 0x20]
   1fcee: 05 47        	<unknown>
   1fcf0: 00 00 00 00  	v.nop
   1fcf4: 00 00 00 00 00 00    	m.nop
   1fcfa: f1 9c 4c 00  	l.ld.w r19, [r19, 0xc]
   1fcfe: 01 4e        	<unknown>
   1fd00: 00 00 00 00  	v.nop
   1fd04: 00 00 00 00 00 00    	m.nop
   1fd0a: 71 15 40 06  	l.ld.w r21, [sp, 0x60]
   1fd0e: 01 00        	c.nop
   1fd10: 00 00 00 00  	v.nop
   1fd14: 00 00 00 00 00 00    	m.nop
   1fd1a: f1 11 44 04  	l.ld.w r7, [sp, 0x44]
   1fd1e: 01 00        	c.nop
   1fd20: 00 00 00 00  	v.nop
   1fd24: 00 00 00 00 00 00    	m.nop
   1fd2a: a3 26 9f 03  	l.srli r26, r4, 0x1f
   1fd2e: 01 00        	c.nop
   1fd30: 00 00 00 00  	v.nop
   1fd34: 00 00 00 00 00 00    	m.nop
   1fd3a: 31 35 c5 12  	l.st.w r5, [r6, 0x134]
   1fd3e: 01 00        	c.nop
   1fd40: 00 00 00 00  	v.nop
   1fd44: 00 00 00 00 00 00    	m.nop
   1fd4a: 2f 28 78 3f  	l.ldi16.s r5, -0x100
   1fd4e: 01 00        	c.nop
   1fd50: 00 00 00 00  	v.nop
   1fd54: 00 00 00 00 00 00    	m.nop
   1fd5a: 31 31 ce 12  	l.st.w r14, [r6, 0x124]
   1fd5e: 01 00        	c.nop
   1fd60: 00 00 00 00  	v.nop
   1fd64: 00 00 00 00 00 00    	m.nop
   1fd6a: 31 36 ce 12  	l.st.w r14, [r6, 0x138]
   1fd6e: 01 00        	c.nop
   1fd70: 00 00 00 00  	v.nop
   1fd74: 00 00 00 00 00 00    	m.nop
   1fd7a: 31 33 c0 14  	l.st.w r0, [r6, 0x14c]
   1fd7e: 01 00        	c.nop
   1fd80: 00 00 00 00  	v.nop
   1fd84: 00 00 00 00 00 00    	m.nop
   1fd8a: 31 35 c0 16  	l.st.w r0, [r6, 0x174]
   1fd8e: 01 00        	c.nop
   1fd90: 00 00 00 00  	v.nop
   1fd94: 00 00 00 00 00 00    	m.nop
   1fd9a: 31 32 ce 12  	l.st.w r14, [r6, 0x128]
   1fd9e: 01 00        	c.nop
   1fda0: 00 00 00 00  	v.nop
   1fda4: 00 00 00 00 00 00    	m.nop
   1fdaa: 31 37 ce 12  	l.st.w r14, [r6, 0x13c]
   1fdae: 01 00        	c.nop
   1fdb0: 00 00 00 00  	v.nop
   1fdb4: 00 00 00 00 00 00    	m.nop
   1fdba: 31 34 c0 14  	l.st.w r0, [r6, 0x150]
   1fdbe: 01 00        	c.nop
   1fdc0: 00 00 00 00  	v.nop
   1fdc4: 00 00 00 00 00 00    	m.nop
   1fdca: 31 36 c0 16  	l.st.w r0, [r6, 0x178]
   1fdce: 01 00        	c.nop
   1fdd0: 00 00 00 00  	v.nop
   1fdd4: 00 00 00 00 00 00    	m.nop
   1fdda: 31 33 ce 12  	l.st.w r14, [r6, 0x12c]
   1fdde: 01 00        	c.nop
   1fde0: 00 00 00 00  	v.nop
   1fde4: 00 00 00 00 00 00    	m.nop
   1fdea: 31 30 ce 14  	l.st.w r14, [r6, 0x140]
   1fdee: 01 00        	c.nop
   1fdf0: 00 00 00 00  	v.nop
   1fdf4: 00 00 00 00 00 00    	m.nop
   1fdfa: 31 35 c0 14  	l.st.w r0, [r6, 0x154]
   1fdfe: 01 00        	c.nop
   1fe00: 00 00 00 00  	v.nop
   1fe04: 00 00 00 00 00 00    	m.nop
   1fe0a: 31 37 c0 16  	l.st.w r0, [r6, 0x17c]
   1fe0e: 01 00        	c.nop
   1fe10: 00 00 00 00  	v.nop
   1fe14: 00 00 00 00 00 00    	m.nop
   1fe1a: 31 34 ce 12  	l.st.w r14, [r6, 0x130]
   1fe1e: 01 00        	c.nop
   1fe20: 00 00 00 00  	v.nop
   1fe24: 00 00 00 00 00 00    	m.nop
   1fe2a: 31 31 ce 14  	l.st.w r14, [r6, 0x144]
   1fe2e: 01 00        	c.nop
   1fe30: 00 00 00 00  	v.nop
   1fe34: 00 00 00 00 00 00    	m.nop
   1fe3a: 31 36 c0 14  	l.st.w r0, [r6, 0x158]
   1fe3e: 01 00        	c.nop
   1fe40: 00 00 00 00  	v.nop
   1fe44: 00 00 00 00 00 00    	m.nop
   1fe4a: 31 30 c0 18  	l.st.w r0, [r6, 0x180]
   1fe4e: 01 00        	c.nop
   1fe50: 00 00 00 00  	v.nop
   1fe54: 00 00 00 00 00 00    	m.nop
   1fe5a: 31 32 d3 14  	l.st.w r19, [r6, 0x148]
   1fe5e: 01 00        	c.nop
   1fe60: 00 00 00 00  	v.nop
   1fe64: 00 00 00 00 00 00    	m.nop
   1fe6a: 31 37 c4 14  	l.st.w r4, [r6, 0x15c]
   1fe6e: 01 00        	c.nop
   1fe70: 00 00 00 00  	v.nop
   1fe74: 00 00 00 00 00 00    	m.nop
   1fe7a: 31 31 da 18  	l.st.w r26, [r6, 0x184]
   1fe7e: 01 00        	c.nop
   1fe80: 00 00 00 00  	v.nop
   1fe84: 00 00 00 00 00 00    	m.nop
   1fe8a: bb 38 85 c0  	l.blt.s r7, r5, 0x2 <0x1fea0>
   1fe8e: 01 00        	c.nop
   1fe90: 00 00 00 00  	v.nop
   1fe94: 00 00 00 00 00 00    	m.nop
   1fe9a: 00 00        	s.nop16
   1fe9c: 00 00        	s.nop16
   1fe9e: 92 8a        	<unknown>
   1fea0: 00 00 00 00  	v.nop
   1fea4: 00 00 00 00 00 00    	m.nop
   1feaa: ef ef 7f 1f  	l.ldi16.u r29, 0xffff
   1feae: ba 86        	<unknown>
   1feb0: 00 00 00 00  	v.nop
   1feb4: 00 00 00 00 00 00    	m.nop
   1feba: f1 15 4c 08  	l.ld.w r23, [sp, 0x8c]
   1febe: 72 89        	<unknown>
   1fec0: 00 00 00 00  	v.nop
   1fec4: 00 00 00 00 00 00    	m.nop
   1feca: ef ef 87 00  	l.ldi_hi r29, 0xff
   1fece: 3a 8a        	<unknown>
   1fed0: 00 00 00 00  	v.nop
   1fed4: 00 00 00 00 00 00    	m.nop
   1feda: 00 00        	s.nop16
   1fedc: 00 00        	s.nop16
   1fede: ba 87        	<unknown>
   1fee0: 00 00 00 00  	v.nop
   1fee4: 00 00 00 00 00 00    	m.nop
   1feea: 00 00        	s.nop16
   1feec: 00 00        	s.nop16
   1feee: f2 83        	<unknown>
   1fef0: 00 00 00 00  	v.nop
   1fef4: 00 00 00 00 00 00    	m.nop
   1fefa: 00 00        	s.nop16
   1fefc: 00 00        	s.nop16
   1fefe: 3a 88        	<unknown>
   1ff00: 00 00 00 00  	v.nop
   1ff04: 00 00 00 00 00 00    	m.nop
   1ff0a: 00 00        	s.nop16
   1ff0c: 00 00        	s.nop16
   1ff0e: ba 8d        	<unknown>
   1ff10: 00 00 00 00  	v.nop
   1ff14: 00 00 00 00 00 00    	m.nop
   1ff1a: 00 00        	s.nop16
   1ff1c: 00 00        	s.nop16
   1ff1e: f2 88        	<unknown>
   1ff20: 00 00 00 00  	v.nop
   1ff24: 00 00 00 00 00 00    	m.nop
   1ff2a: 00 00        	s.nop16
   1ff2c: 00 00        	s.nop16
   1ff2e: ba 8c        	<unknown>
   1ff30: 00 00 00 00  	v.nop
   1ff34: 00 00 00 00 00 00    	m.nop
   1ff3a: 00 00        	s.nop16
   1ff3c: 00 00        	s.nop16
   1ff3e: 72 8b        	<unknown>
   1ff40: 00 00 00 00  	v.nop
   1ff44: 00 00 00 00 00 00    	m.nop
   1ff4a: 00 00        	s.nop16
   1ff4c: 00 00        	s.nop16
   1ff4e: f2 82        	<unknown>
   1ff50: 00 00 00 00  	v.nop
   1ff54: 00 00 00 00 00 00    	m.nop
   1ff5a: 2b 10 8a 01  	l.jal r0, 0xa2 <0x20970>
   1ff5e: 01 00        	c.nop
   1ff60: 00 00 00 00  	v.nop
   1ff64: 00 00 00 00 00 00    	m.nop
   1ff6a: 7b 28 60 00  	<unknown>
   1ff6e: 01 00        	c.nop
   1ff70: 00 00 00 00  	v.nop
   1ff74: 00 00 00 00 00 00    	m.nop
   1ff7a: 7b 28 1a 41  	<unknown>
   1ff7e: 01 00        	c.nop
   1ff80: 00 00 00 00  	v.nop
   1ff84: 00 00 00 00 00 00    	m.nop
   1ff8a: 71 11 54 02  	l.ld.w r5, [sp, 0x34]
   1ff8e: 3e 85        	<unknown>
   1ff90: 00 00 00 00  	v.nop
   1ff94: 00 00 00 00 00 00    	m.nop
   1ff9a: f1 15 50 02  	l.ld.w r23, [sp, 0x30]
   1ff9e: 12 84        	<unknown>
   1ffa0: 00 00 00 00  	v.nop
   1ffa4: 00 00 00 00 00 00    	m.nop
   1ffaa: 31 16 58 02  	l.ld.w r24, [sp, 0x38]
   1ffae: 01 00        	c.nop
   1ffb0: 00 00 00 00  	v.nop
   1ffb4: 00 00 00 00 00 00    	m.nop
   1ffba: 71 16 40 04  	l.ld.w r25, [sp, 0x40]
   1ffbe: 01 00        	c.nop
   1ffc0: 00 00 00 00  	v.nop
   1ffc4: 00 00 00 00 00 00    	m.nop
   1ffca: 71 15 48 02  	l.ld.w r21, [sp, 0x28]
   1ffce: 01 00        	c.nop
   1ffd0: 00 00 00 00  	v.nop
   1ffd4: 00 00 00 00 00 00    	m.nop
   1ffda: f1 12 40 02  	l.ld.w r11, [sp, 0x20]
   1ffde: 16 8e        	<unknown>
   1ffe0: 00 00 00 00  	v.nop
   1ffe4: 00 00 00 00 00 00    	m.nop
   1ffea: f1 9e 4c 00  	l.ld.w r27, [r19, 0xc]
   1ffee: 5e 87        	<unknown>
   1fff0: 00 00 00 00  	v.nop
   1fff4: 00 00 00 00 00 00    	m.nop
   1fffa: f1 9b 50 00  	l.ld.w r15, [r19, 0x10]
   1fffe: e2 83        	<unknown>
   20000: 00 00 00 00  	v.nop
   20004: 00 00 00 00 00 00    	m.nop
   2000a: 71 9b 54 00  	l.ld.w r13, [r19, 0x14]
   2000e: 66 89        	<unknown>
   20010: 00 00 00 00  	v.nop
   20014: 00 00 00 00 00 00    	m.nop
   2001a: 63 24 9f 03  	l.srli r17, r4, 0x1f
   2001e: 56 8a        	<unknown>
   20020: 00 00 00 00  	v.nop
   20024: 00 00 00 00 00 00    	m.nop
   2002a: 31 36 d8 14  	l.st.w r24, [r6, 0x158]
   2002e: 2e 88        	<unknown>
   20030: 00 00 00 00  	v.nop
   20034: 00 00 00 00 00 00    	m.nop
   2003a: 31 35 c5 12  	l.st.w r5, [r6, 0x134]
   2003e: 72 8c        	<unknown>
   20040: 00 00 00 00  	v.nop
   20044: 00 00 00 00 00 00    	m.nop
   2004a: b1 15 40 10  	l.ld.w r22, [sp, 0x100]
   2004e: 01 00        	c.nop
   20050: 00 00 00 00  	v.nop
   20054: 00 00 00 00 00 00    	m.nop
   2005a: 31 31 da 12  	l.st.w r26, [r6, 0x124]
   2005e: 01 00        	c.nop
   20060: 00 00 00 00  	v.nop
   20064: 00 00 00 00 00 00    	m.nop
   2006a: 31 36 da 12  	l.st.w r26, [r6, 0x138]
   2006e: 01 00        	c.nop
   20070: 00 00 00 00  	v.nop
   20074: 00 00 00 00 00 00    	m.nop
   2007a: 31 33 ca 14  	l.st.w r10, [r6, 0x14c]
   2007e: 01 00        	c.nop
   20080: 00 00 00 00  	v.nop
   20084: 00 00 00 00 00 00    	m.nop
   2008a: 31 35 ca 16  	l.st.w r10, [r6, 0x174]
   2008e: 01 00        	c.nop
   20090: 00 00 00 00  	v.nop
   20094: 00 00 00 00 00 00    	m.nop
   2009a: 31 32 cb 12  	l.st.w r11, [r6, 0x128]
   2009e: 01 00        	c.nop
   200a0: 00 00 00 00  	v.nop
   200a4: 00 00 00 00 00 00    	m.nop
   200aa: 31 37 db 12  	l.st.w r27, [r6, 0x13c]
   200ae: 01 00        	c.nop
   200b0: 00 00 00 00  	v.nop
   200b4: 00 00 00 00 00 00    	m.nop
   200ba: 31 34 c4 14  	l.st.w r4, [r6, 0x150]
   200be: 01 00        	c.nop
   200c0: 00 00 00 00  	v.nop
   200c4: 00 00 00 00 00 00    	m.nop
   200ca: 31 36 d1 16  	l.st.w r17, [r6, 0x178]
   200ce: 01 00        	c.nop
   200d0: 00 00 00 00  	v.nop
   200d4: 00 00 00 00 00 00    	m.nop
   200da: 31 33 d5 12  	l.st.w r21, [r6, 0x12c]
   200de: 01 00        	c.nop
   200e0: 00 00 00 00  	v.nop
   200e4: 00 00 00 00 00 00    	m.nop
   200ea: 31 30 cf 14  	l.st.w r15, [r6, 0x140]
   200ee: 01 00        	c.nop
   200f0: 00 00 00 00  	v.nop
   200f4: 00 00 00 00 00 00    	m.nop
   200fa: 31 35 ca 14  	l.st.w r10, [r6, 0x154]
   200fe: 01 00        	c.nop
   20100: 00 00 00 00  	v.nop
   20104: 00 00 00 00 00 00    	m.nop
   2010a: 31 37 ca 16  	l.st.w r10, [r6, 0x17c]
   2010e: 01 00        	c.nop
   20110: 00 00 00 00  	v.nop
   20114: 00 00 00 00 00 00    	m.nop
   2011a: 31 34 d7 12  	l.st.w r23, [r6, 0x130]
   2011e: 01 00        	c.nop
   20120: 00 00 00 00  	v.nop
   20124: 00 00 00 00 00 00    	m.nop
   2012a: 31 31 cd 14  	l.st.w r13, [r6, 0x144]
   2012e: 01 00        	c.nop
   20130: 00 00 00 00  	v.nop
   20134: 00 00 00 00 00 00    	m.nop
   2013a: 31 30 d9 18  	l.st.w r25, [r6, 0x180]
   2013e: 01 00        	c.nop
   20140: 00 00 00 00  	v.nop
   20144: 00 00 00 00 00 00    	m.nop
   2014a: 61 99 18 00  	l.addi.u r5, r19, 0x18
   2014e: 01 00        	c.nop
   20150: 00 00 00 00  	v.nop
   20154: 00 00 00 00 00 00    	m.nop
   2015a: 2b 38 83 01  	l.jal r0, 0x37 <0x204c0>
   2015e: 01 00        	c.nop
   20160: 00 00 00 00  	v.nop
   20164: 00 00 00 00 00 00    	m.nop
   2016a: 71 11 40 02  	l.ld.w r5, [sp, 0x20]
   2016e: 05 48        	<unknown>
   20170: 00 00 00 00  	v.nop
   20174: 00 00 00 00 00 00    	m.nop
   2017a: f1 12 50 02  	l.ld.w r11, [sp, 0x30]
   2017e: 01 49        	<unknown>
   20180: 00 00 00 00  	v.nop
   20184: 00 00 00 00 00 00    	m.nop
   2018a: f1 15 40 04  	l.ld.w r23, [sp, 0x40]
   2018e: 12 8e        	<unknown>
   20190: 00 00 00 00  	v.nop
   20194: 00 00 00 00 00 00    	m.nop
   2019a: b1 12 48 02  	l.ld.w r10, [sp, 0x28]
   2019e: c2 8d        	<unknown>
   201a0: 00 00 00 00  	v.nop
   201a4: 00 00 00 00 00 00    	m.nop
   201aa: f1 9b 4c 00  	l.ld.w r15, [r19, 0xc]
   201ae: 4a 84        	<unknown>
   201b0: 00 00 00 00  	v.nop
   201b4: 00 00 00 00 00 00    	m.nop
   201ba: 71 9b 50 00  	l.ld.w r13, [r19, 0x10]
   201be: 16 8a        	<unknown>
   201c0: 00 00 00 00  	v.nop
   201c4: 00 00 00 00 00 00    	m.nop
   201ca: f1 9c 54 00  	l.ld.w r19, [r19, 0x14]
   201ce: 2e 8c        	<unknown>
   201d0: 00 00 00 00  	v.nop
   201d4: 00 00 00 00 00 00    	m.nop
   201da: 31 33 c5 12  	l.st.w r5, [r6, 0x12c]
   201de: 5e 8d        	<unknown>
   201e0: 00 00 00 00  	v.nop
   201e4: 00 00 00 00 00 00    	m.nop
   201ea: 71 11 58 02  	l.ld.w r5, [sp, 0x38]
   201ee: 2a 87        	<unknown>
   201f0: 00 00 00 00  	v.nop
   201f4: 00 00 00 00 00 00    	m.nop
   201fa: e3 21 9f 03  	l.srli r7, r4, 0x1f
   201fe: ca 88        	<unknown>
   20200: 00 00 00 00  	v.nop
   20204: 00 00 00 00 00 00    	m.nop
   2020a: 71 15 48 08  	l.ld.w r21, [sp, 0x88]
   2020e: c2 8c        	<unknown>
   20210: 00 00 00 00  	v.nop
   20214: 00 00 00 00 00 00    	m.nop
   2021a: 31 31 d0 12  	l.st.w r16, [r6, 0x124]
   2021e: 4a 8b        	<unknown>
   20220: 00 00 00 00  	v.nop
   20224: 00 00 00 00 00 00    	m.nop
   2022a: 31 36 d0 12  	l.st.w r16, [r6, 0x138]
   2022e: 01 00        	c.nop
   20230: 00 00 00 00  	v.nop
   20234: 00 00 00 00 00 00    	m.nop
   2023a: 31 33 c0 14  	l.st.w r0, [r6, 0x14c]
   2023e: 01 00        	c.nop
   20240: 00 00 00 00  	v.nop
   20244: 00 00 00 00 00 00    	m.nop
   2024a: 31 35 c0 16  	l.st.w r0, [r6, 0x174]
   2024e: 01 00        	c.nop
   20250: 00 00 00 00  	v.nop
   20254: 00 00 00 00 00 00    	m.nop
   2025a: 31 32 d0 12  	l.st.w r16, [r6, 0x128]
   2025e: 01 00        	c.nop
   20260: 00 00 00 00  	v.nop
   20264: 00 00 00 00 00 00    	m.nop
   2026a: 31 37 d0 12  	l.st.w r16, [r6, 0x13c]
   2026e: 01 00        	c.nop
   20270: 00 00 00 00  	v.nop
   20274: 00 00 00 00 00 00    	m.nop
   2027a: 31 34 c0 14  	l.st.w r0, [r6, 0x150]
   2027e: 01 00        	c.nop
   20280: 00 00 00 00  	v.nop
   20284: 00 00 00 00 00 00    	m.nop
   2028a: 31 36 c0 16  	l.st.w r0, [r6, 0x178]
   2028e: 01 00        	c.nop
   20290: 00 00 00 00  	v.nop
   20294: 00 00 00 00 00 00    	m.nop
   2029a: 31 30 cf 14  	l.st.w r15, [r6, 0x140]
   2029e: 01 00        	c.nop
   202a0: 00 00 00 00  	v.nop
   202a4: 00 00 00 00 00 00    	m.nop
   202aa: 31 35 c4 14  	l.st.w r4, [r6, 0x154]
   202ae: 01 00        	c.nop
   202b0: 00 00 00 00  	v.nop
   202b4: 00 00 00 00 00 00    	m.nop
   202ba: 31 37 c7 16  	l.st.w r7, [r6, 0x17c]
   202be: 01 00        	c.nop
   202c0: 00 00 00 00  	v.nop
   202c4: 00 00 00 00 00 00    	m.nop
   202ca: 31 34 ca 12  	l.st.w r10, [r6, 0x130]
   202ce: 01 00        	c.nop
   202d0: 00 00 00 00  	v.nop
   202d4: 00 00 00 00 00 00    	m.nop
   202da: 31 31 cd 14  	l.st.w r13, [r6, 0x144]
   202de: 01 00        	c.nop
   202e0: 00 00 00 00  	v.nop
   202e4: 00 00 00 00 00 00    	m.nop
   202ea: 31 36 c0 14  	l.st.w r0, [r6, 0x158]
   202ee: 01 00        	c.nop
   202f0: 00 00 00 00  	v.nop
   202f4: 00 00 00 00 00 00    	m.nop
   202fa: 31 30 c0 18  	l.st.w r0, [r6, 0x180]
   202fe: 01 00        	c.nop
   20300: 00 00 00 00  	v.nop
   20304: 00 00 00 00 00 00    	m.nop
   2030a: 31 35 cb 12  	l.st.w r11, [r6, 0x134]
   2030e: 01 00        	c.nop
   20310: 00 00 00 00  	v.nop
   20314: 00 00 00 00 00 00    	m.nop
   2031a: 31 32 d3 14  	l.st.w r19, [r6, 0x148]
   2031e: 01 00        	c.nop
   20320: 00 00 00 00  	v.nop
   20324: 00 00 00 00 00 00    	m.nop
   2032a: 31 37 c5 14  	l.st.w r5, [r6, 0x15c]
   2032e: 01 00        	c.nop
   20330: 00 00 00 00  	v.nop
   20334: 00 00 00 00 00 00    	m.nop
   2033a: 31 31 d7 18  	l.st.w r23, [r6, 0x184]
   2033e: 01 00        	c.nop
   20340: 00 00 00 00  	v.nop
   20344: 00 00 00 00 00 00    	m.nop
   2034a: 2b 50 85 01  	l.jal r0, 0x5a <0x208e0>
   2034e: 01 00        	c.nop
   20350: 00 00 00 00  	v.nop
   20354: 00 00 00 00 00 00    	m.nop
   2035a: b1 12 48 02  	l.ld.w r10, [sp, 0x28]
   2035e: 05 4a        	<unknown>
   20360: 00 00 00 00  	v.nop
   20364: 00 00 00 00 00 00    	m.nop
   2036a: 71 11 40 02  	l.ld.w r5, [sp, 0x20]
   2036e: 01 44        	<unknown>
   20370: 00 00 00 00  	v.nop
   20374: 00 00 00 00 00 00    	m.nop
   2037a: 71 9b 4c 00  	l.ld.w r13, [r19, 0xc]
   2037e: 92 83        	<unknown>
   20380: 00 00 00 00  	v.nop
   20384: 00 00 00 00 00 00    	m.nop
   2038a: b1 15 40 06  	l.ld.w r22, [sp, 0x60]
   2038e: d2 87        	<unknown>
   20390: 00 00 00 00  	v.nop
   20394: 00 00 00 00 00 00    	m.nop
   2039a: a3 24 9f 03  	l.srli r18, r4, 0x1f
   2039e: 52 88        	<unknown>
   203a0: 00 00 00 00  	v.nop
   203a4: 00 00 00 00 00 00    	m.nop
   203aa: 31 31 d4 12  	l.st.w r20, [r6, 0x124]
   203ae: 2a 8c        	<unknown>
   203b0: 00 00 00 00  	v.nop
   203b4: 00 00 00 00 00 00    	m.nop
   203ba: 31 34 c5 12  	l.st.w r5, [r6, 0x130]
   203be: 16 87        	<unknown>
   203c0: 00 00 00 00  	v.nop
   203c4: 00 00 00 00 00 00    	m.nop
   203ca: 61 99 10 00  	l.addi.u r5, r19, 0x10
   203ce: d2 8d        	<unknown>
   203d0: 00 00 00 00  	v.nop
   203d4: 00 00 00 00 00 00    	m.nop
   203da: 31 36 d4 12  	l.st.w r20, [r6, 0x138]
   203de: a2 88        	<unknown>
   203e0: 00 00 00 00  	v.nop
   203e4: 00 00 00 00 00 00    	m.nop
   203ea: 31 33 c0 14  	l.st.w r0, [r6, 0x14c]
   203ee: 01 00        	c.nop
   203f0: 00 00 00 00  	v.nop
   203f4: 00 00 00 00 00 00    	m.nop
   203fa: 31 35 c0 16  	l.st.w r0, [r6, 0x174]
   203fe: 01 00        	c.nop
   20400: 00 00 00 00  	v.nop
   20404: 00 00 00 00 00 00    	m.nop
   2040a: 31 32 d4 12  	l.st.w r20, [r6, 0x128]
   2040e: 01 00        	c.nop
   20410: 00 00 00 00  	v.nop
   20414: 00 00 00 00 00 00    	m.nop
   2041a: 31 37 d4 12  	l.st.w r20, [r6, 0x13c]
   2041e: 01 00        	c.nop
   20420: 00 00 00 00  	v.nop
   20424: 00 00 00 00 00 00    	m.nop
   2042a: 31 34 c0 14  	l.st.w r0, [r6, 0x150]
   2042e: 01 00        	c.nop
   20430: 00 00 00 00  	v.nop
   20434: 00 00 00 00 00 00    	m.nop
   2043a: 31 36 c0 16  	l.st.w r0, [r6, 0x178]
   2043e: 01 00        	c.nop
   20440: 00 00 00 00  	v.nop
   20444: 00 00 00 00 00 00    	m.nop
   2044a: 31 33 d4 12  	l.st.w r20, [r6, 0x12c]
   2044e: 01 00        	c.nop
   20450: 00 00 00 00  	v.nop
   20454: 00 00 00 00 00 00    	m.nop
   2045a: 31 30 d4 14  	l.st.w r20, [r6, 0x140]
   2045e: 01 00        	c.nop
   20460: 00 00 00 00  	v.nop
   20464: 00 00 00 00 00 00    	m.nop
   2046a: 31 35 c0 14  	l.st.w r0, [r6, 0x154]
   2046e: 01 00        	c.nop
   20470: 00 00 00 00  	v.nop
   20474: 00 00 00 00 00 00    	m.nop
   2047a: 31 37 c0 16  	l.st.w r0, [r6, 0x17c]
   2047e: 01 00        	c.nop
   20480: 00 00 00 00  	v.nop
   20484: 00 00 00 00 00 00    	m.nop
   2048a: 31 31 cd 14  	l.st.w r13, [r6, 0x144]
   2048e: 01 00        	c.nop
   20490: 00 00 00 00  	v.nop
   20494: 00 00 00 00 00 00    	m.nop
   2049a: 31 36 c4 14  	l.st.w r4, [r6, 0x158]
   2049e: 01 00        	c.nop
   204a0: 00 00 00 00  	v.nop
   204a4: 00 00 00 00 00 00    	m.nop
   204aa: 31 30 d2 18  	l.st.w r18, [r6, 0x180]
   204ae: 01 00        	c.nop
   204b0: 00 00 00 00  	v.nop
   204b4: 00 00 00 00 00 00    	m.nop
   204ba: 31 35 ca 12  	l.st.w r10, [r6, 0x134]
   204be: 01 00        	c.nop
   204c0: 00 00 00 00  	v.nop
   204c4: 00 00 00 00 00 00    	m.nop
   204ca: f1 2c 40 00  	l.ld.w r19, [r5]
   204ce: 81 4a        	<unknown>
   204d0: 00 00 00 00  	v.nop
   204d4: 00 00 00 00 00 00    	m.nop
   204da: 31 37 c0 14  	l.st.w r0, [r6, 0x15c]
   204de: 01 00        	c.nop
   204e0: 00 00 00 00  	v.nop
   204e4: 00 00 00 00 00 00    	m.nop
   204ea: 31 31 c0 18  	l.st.w r0, [r6, 0x184]
   204ee: 01 00        	c.nop
   204f0: 00 00 00 00  	v.nop
   204f4: 00 00 00 00 00 00    	m.nop
   204fa: 31 32 d3 14  	l.st.w r19, [r6, 0x148]
   204fe: 01 00        	c.nop
   20500: 00 00 00 00  	v.nop
   20504: 00 00 00 00 00 00    	m.nop
   2050a: bb 38 80 c0  	l.blt.s r7, r0, 0x2 <0x20520>
   2050e: 01 00        	c.nop
   20510: 00 00 00 00  	v.nop
   20514: 00 00 00 00 00 00    	m.nop
   2051a: 00 00        	s.nop16
   2051c: 00 00        	s.nop16
   2051e: 1e 8b        	<unknown>
   20520: 00 00 00 00  	v.nop
   20524: 00 00 00 00 00 00    	m.nop
   2052a: 00 00        	s.nop16
   2052c: 00 00        	s.nop16
   2052e: 85 4c        	<unknown>
   20530: 00 00 00 00  	v.nop
   20534: 00 00 00 00 00 00    	m.nop
   2053a: 00 00        	s.nop16
   2053c: 00 00        	s.nop16
   2053e: d6 83        	<unknown>
   20540: 00 00 00 00  	v.nop
   20544: 00 00 00 00 00 00    	m.nop
   2054a: 00 00        	s.nop16
   2054c: 00 00        	s.nop16
   2054e: 56 8d        	<unknown>
   20550: 00 00 00 00  	v.nop
   20554: 00 00 00 00 00 00    	m.nop
   2055a: 00 00        	s.nop16
   2055c: 00 00        	s.nop16
   2055e: 56 8e        	<unknown>
   20560: 00 00 00 00  	v.nop
   20564: 00 00 00 00 00 00    	m.nop
   2056a: 2b 58 83 01  	l.jal r0, 0x3b <0x20910>
   2056e: 01 00        	c.nop
   20570: 00 00 00 00  	v.nop
   20574: 00 00 00 00 00 00    	m.nop
   2057a: 31 16 48 02  	l.ld.w r24, [sp, 0x28]
   2057e: 01 4b        	<unknown>
   20580: 00 00 00 00  	v.nop
   20584: 00 00 00 00 00 00    	m.nop
   2058a: 23 22 9f 03  	l.srli r8, r4, 0x1f
   2058e: 5a 8d        	<unknown>
   20590: 00 00 00 00  	v.nop
   20594: 00 00 00 00 00 00    	m.nop
   2059a: b1 12 54 02  	l.ld.w r10, [sp, 0x34]
   2059e: 5a 89        	<unknown>
   205a0: 00 00 00 00  	v.nop
   205a4: 00 00 00 00 00 00    	m.nop
   205aa: f1 15 50 02  	l.ld.w r23, [sp, 0x30]
   205ae: da 88        	<unknown>
   205b0: 00 00 00 00  	v.nop
   205b4: 00 00 00 00 00 00    	m.nop
   205ba: 71 17 40 04  	l.ld.w r29, [sp, 0x40]
   205be: 01 00        	c.nop
   205c0: 00 00 00 00  	v.nop
   205c4: 00 00 00 00 00 00    	m.nop
   205ca: f1 12 40 02  	l.ld.w r11, [sp, 0x20]
   205ce: 62 88        	<unknown>
   205d0: 00 00 00 00  	v.nop
   205d4: 00 00 00 00 00 00    	m.nop
   205da: 71 99 4c 00  	l.ld.w r5, [r19, 0xc]
   205de: 01 00        	c.nop
   205e0: 00 00 00 00  	v.nop
   205e4: 00 00 00 00 00 00    	m.nop
   205ea: f1 9e 50 00  	l.ld.w r27, [r19, 0x10]
   205ee: 2a 87        	<unknown>
   205f0: 00 00 00 00  	v.nop
   205f4: 00 00 00 00 00 00    	m.nop
   205fa: f1 9b 54 00  	l.ld.w r15, [r19, 0x14]
   205fe: 5e 8a        	<unknown>
   20600: 00 00 00 00  	v.nop
   20604: 00 00 00 00 00 00    	m.nop
   2060a: 71 9b 58 00  	l.ld.w r13, [r19, 0x18]
   2060e: f6 83        	<unknown>
   20610: 00 00 00 00  	v.nop
   20614: 00 00 00 00 00 00    	m.nop
   2061a: f1 9c 5c 00  	l.ld.w r19, [r19, 0x1c]
   2061e: ae 8c        	<unknown>
   20620: 00 00 00 00  	v.nop
   20624: 00 00 00 00 00 00    	m.nop
   2062a: 31 35 c8 16  	l.st.w r8, [r6, 0x174]
   2062e: 01 00        	c.nop
   20630: 00 00 00 00  	v.nop
   20634: 00 00 00 00 00 00    	m.nop
   2063a: 31 12 58 02  	l.ld.w r8, [sp, 0x38]
   2063e: 01 00        	c.nop
   20640: 00 00 00 00  	v.nop
   20644: 00 00 00 00 00 00    	m.nop
   2064a: 31 32 d8 12  	l.st.w r24, [r6, 0x128]
   2064e: 05 4c        	<unknown>
   20650: 00 00 00 00  	v.nop
   20654: 00 00 00 00 00 00    	m.nop
   2065a: 31 36 c5 12  	l.st.w r5, [r6, 0x138]
   2065e: da 82        	<unknown>
   20660: 00 00 00 00  	v.nop
   20664: 00 00 00 00 00 00    	m.nop
   2066a: 31 17 48 08  	l.ld.w r28, [sp, 0x88]
   2066e: 01 00        	c.nop
   20670: 00 00 00 00  	v.nop
   20674: 00 00 00 00 00 00    	m.nop
   2067a: 31 31 cb 12  	l.st.w r11, [r6, 0x124]
   2067e: 01 00        	c.nop
   20680: 00 00 00 00  	v.nop
   20684: 00 00 00 00 00 00    	m.nop
   2068a: 31 33 c4 14  	l.st.w r4, [r6, 0x14c]
   2068e: 01 00        	c.nop
   20690: 00 00 00 00  	v.nop
   20694: 00 00 00 00 00 00    	m.nop
   2069a: 31 37 db 12  	l.st.w r27, [r6, 0x13c]
   2069e: 01 00        	c.nop
   206a0: 00 00 00 00  	v.nop
   206a4: 00 00 00 00 00 00    	m.nop
   206aa: 31 34 c0 14  	l.st.w r0, [r6, 0x150]
   206ae: 01 00        	c.nop
   206b0: 00 00 00 00  	v.nop
   206b4: 00 00 00 00 00 00    	m.nop
   206ba: 31 36 c0 16  	l.st.w r0, [r6, 0x178]
   206be: 01 00        	c.nop
   206c0: 00 00 00 00  	v.nop
   206c4: 00 00 00 00 00 00    	m.nop
   206ca: 31 33 d7 12  	l.st.w r23, [r6, 0x12c]
   206ce: 01 00        	c.nop
   206d0: 00 00 00 00  	v.nop
   206d4: 00 00 00 00 00 00    	m.nop
   206da: 31 30 cf 14  	l.st.w r15, [r6, 0x140]
   206de: 01 00        	c.nop
   206e0: 00 00 00 00  	v.nop
   206e4: 00 00 00 00 00 00    	m.nop
   206ea: 31 35 c8 14  	l.st.w r8, [r6, 0x154]
   206ee: 01 00        	c.nop
   206f0: 00 00 00 00  	v.nop
   206f4: 00 00 00 00 00 00    	m.nop
   206fa: 31 37 dd 16  	l.st.w r29, [r6, 0x17c]
   206fe: 01 00        	c.nop
   20700: 00 00 00 00  	v.nop
   20704: 00 00 00 00 00 00    	m.nop
   2070a: 31 34 ca 12  	l.st.w r10, [r6, 0x130]
   2070e: 01 00        	c.nop
   20710: 00 00 00 00  	v.nop
   20714: 00 00 00 00 00 00    	m.nop
   2071a: 31 31 cd 14  	l.st.w r13, [r6, 0x144]
   2071e: 01 00        	c.nop
   20720: 00 00 00 00  	v.nop
   20724: 00 00 00 00 00 00    	m.nop
   2072a: 31 36 c0 14  	l.st.w r0, [r6, 0x158]
   2072e: 01 00        	c.nop
   20730: 00 00 00 00  	v.nop
   20734: 00 00 00 00 00 00    	m.nop
   2073a: 31 30 c0 18  	l.st.w r0, [r6, 0x180]
   2073e: 01 00        	c.nop
   20740: 00 00 00 00  	v.nop
   20744: 00 00 00 00 00 00    	m.nop
   2074a: 31 35 d8 12  	l.st.w r24, [r6, 0x134]
   2074e: 01 00        	c.nop
   20750: 00 00 00 00  	v.nop
   20754: 00 00 00 00 00 00    	m.nop
   2075a: 31 32 d3 14  	l.st.w r19, [r6, 0x148]
   2075e: 01 00        	c.nop
   20760: 00 00 00 00  	v.nop
   20764: 00 00 00 00 00 00    	m.nop
   2076a: 31 31 c0 18  	l.st.w r0, [r6, 0x184]
   2076e: 01 00        	c.nop
   20770: 00 00 00 00  	v.nop
   20774: 00 00 00 00 00 00    	m.nop
   2077a: 2b 68 81 01  	l.jal r0, 0x1d <0x20940>
   2077e: 01 00        	c.nop
   20780: 00 00 00 00  	v.nop
   20784: 00 00 00 00 00 00    	m.nop
   2078a: f1 31 58 14  	l.ld.w r7, [r6, 0x158]
   2078e: 01 00        	c.nop
   20790: 00 00 00 00  	v.nop
   20794: 00 00 00 00 00 00    	m.nop
   2079a: 71 31 5c 14  	l.ld.w r5, [r6, 0x15c]
   2079e: 01 00        	c.nop
   207a0: 00 00 00 00  	v.nop
   207a4: 00 00 00 00 00 00    	m.nop
   207aa: 31 36 54 12  	l.ld.w r24, [r6, 0x134]
   207ae: 01 00        	c.nop
   207b0: 00 00 00 00  	v.nop
   207b4: 00 00 00 00 00 00    	m.nop
   207ba: f1 34 48 14  	l.ld.w r19, [r6, 0x148]
   207be: 01 00        	c.nop
   207c0: 00 00 00 00  	v.nop
   207c4: 00 00 00 00 00 00    	m.nop
   207ca: b1 36 44 18  	l.ld.w r26, [r6, 0x184]
   207ce: 01 00        	c.nop
   207d0: 00 00 00 00  	v.nop
   207d4: 00 00 00 00 00 00    	m.nop
   207da: a3 3d 9f 0b  	l.srai r22, r7, 0x1f
   207de: 01 00        	c.nop
   207e0: 00 00 00 00  	v.nop
   207e4: 00 00 00 00 00 00    	m.nop
   207ea: a3 3a 16 18  	<unknown>
   207ee: 01 00        	c.nop
   207f0: 00 00 00 00  	v.nop
   207f4: 00 00 00 00 00 00    	m.nop
   207fa: b1 33 50 12  	l.ld.w r14, [r6, 0x130]
   207fe: 01 00        	c.nop
   20800: 00 00 00 00  	v.nop
   20804: 00 00 00 00 00 00    	m.nop
   2080a: 71 33 44 14  	l.ld.w r13, [r6, 0x144]
   2080e: 01 00        	c.nop
   20810: 00 00 00 00  	v.nop
   20814: 00 00 00 00 00 00    	m.nop
   2081a: b1 34 40 18  	l.ld.w r18, [r6, 0x180]
   2081e: 01 00        	c.nop
   20820: 00 00 00 00  	v.nop
   20824: 00 00 00 00 00 00    	m.nop
   2082a: 31 35 4c 12  	l.ld.w r20, [r6, 0x12c]
   2082e: 01 00        	c.nop
   20830: 00 00 00 00  	v.nop
   20834: 00 00 00 00 00 00    	m.nop
   2083a: f1 33 40 14  	l.ld.w r15, [r6, 0x140]
   2083e: 01 00        	c.nop
   20840: 00 00 00 00  	v.nop
   20844: 00 00 00 00 00 00    	m.nop
   2084a: 31 37 54 14  	l.ld.w r28, [r6, 0x154]
   2084e: 01 00        	c.nop
   20850: 00 00 00 00  	v.nop
   20854: 00 00 00 00 00 00    	m.nop
   2085a: f1 31 5c 16  	l.ld.w r7, [r6, 0x17c]
   2085e: 01 00        	c.nop
   20860: 00 00 00 00  	v.nop
   20864: 00 00 00 00 00 00    	m.nop
   2086a: 31 34 48 12  	l.ld.w r16, [r6, 0x128]
   2086e: 01 00        	c.nop
   20870: 00 00 00 00  	v.nop
   20874: 00 00 00 00 00 00    	m.nop
   2087a: f1 36 5c 12  	l.ld.w r27, [r6, 0x13c]
   2087e: 01 00        	c.nop
   20880: 00 00 00 00  	v.nop
   20884: 00 00 00 00 00 00    	m.nop
   2088a: 31 32 50 14  	l.ld.w r8, [r6, 0x150]
   2088e: 01 00        	c.nop
   20890: 00 00 00 00  	v.nop
   20894: 00 00 00 00 00 00    	m.nop
   2089a: 71 34 58 16  	l.ld.w r17, [r6, 0x178]
   2089e: 01 00        	c.nop
   208a0: 00 00 00 00  	v.nop
   208a4: 00 00 00 00 00 00    	m.nop
   208aa: 71 36 44 12  	l.ld.w r25, [r6, 0x124]
   208ae: 01 00        	c.nop
   208b0: 00 00 00 00  	v.nop
   208b4: 00 00 00 00 00 00    	m.nop
   208ba: 63 2d 9f 0b  	l.srai r21, r5, 0x1f
   208be: 01 00        	c.nop
   208c0: 00 00 00 00  	v.nop
   208c4: 00 00 00 00 00 00    	m.nop
   208ca: a5 55 96 02  	l.xorix r22, r10, 0x16
   208ce: d6 92        	<unknown>
   208d0: 00 00 00 00  	v.nop
   208d4: 00 00 00 00 00 00    	m.nop
   208da: 65 2d 95 02  	l.xorix r21, r5, 0x15
   208de: 01 00        	c.nop
   208e0: 00 00 00 00  	v.nop
   208e4: 00 00 00 00 00 00    	m.nop
   208ea: 63 e1 9f 0b  	l.srai r5, r28, 0x1f
   208ee: 01 00        	c.nop
   208f0: 00 00 00 00  	v.nop
   208f4: 00 00 00 00 00 00    	m.nop
   208fa: 00 00        	s.nop16
   208fc: 00 00        	s.nop16
   208fe: 16 9e        	<unknown>
   20900: 00 00 00 00  	v.nop
   20904: 00 00 00 00 00 00    	m.nop
   2090a: 25 e7 85 02  	l.xorix r28, r28, 0x5
   2090e: 01 00        	c.nop
   20910: 00 00 00 00  	v.nop
   20914: 00 00 00 00 00 00    	m.nop
   2091a: 63 41 9f 0b  	l.srai r5, r8, 0x1f
   2091e: 01 00        	c.nop
   20920: 00 00 00 00  	v.nop
   20924: 00 00 00 00 00 00    	m.nop
   2092a: 00 00        	s.nop16
   2092c: 00 00        	s.nop16
   2092e: 16 94        	<unknown>
   20930: 00 00 00 00  	v.nop
   20934: 00 00 00 00 00 00    	m.nop
   2093a: 65 41 85 02  	l.xorix r5, r8, 0x5
   2093e: 01 00        	c.nop
   20940: 00 00 00 00  	v.nop
   20944: 00 00 00 00 00 00    	m.nop
   2094a: ef ef 7f 1f  	l.ldi16.u r29, 0xffff
   2094e: 01 00        	c.nop
   20950: 00 00 00 00  	v.nop
   20954: 00 00 00 00 00 00    	m.nop
   2095a: ef ef 87 00  	l.ldi_hi r29, 0xff
   2095e: 01 00        	c.nop
   20960: 00 00 00 00  	v.nop
   20964: 00 00 00 00 00 00    	m.nop
   2096a: e5 c5 9d 01  	l.andix r23, r24, 0x1d
   2096e: 01 00        	c.nop
   20970: 00 00 00 00  	v.nop
   20974: 00 00 00 00 00 00    	m.nop
   2097a: b1 12 4c 02  	l.ld.w r10, [sp, 0x2c]
   2097e: e2 03        	<unknown>
   20980: 00 00 00 00  	v.nop
   20984: 00 00 00 00 00 00    	m.nop
   2098a: a5 73 9d 01  	l.andix r14, r14, 0x1d
   2098e: 62 0d        	<unknown>
   20990: 00 00 00 00  	v.nop
   20994: 00 00 00 00 00 00    	m.nop
   2099a: 25 a5 9d 01  	l.andix r20, r20, 0x1d
   2099e: 62 09        	<unknown>
   209a0: 00 00 00 00  	v.nop
   209a4: 00 00 00 00 00 00    	m.nop
   209aa: 25 84 9d 01  	l.andix r16, r16, 0x1d
   209ae: e2 08        	<unknown>
   209b0: 00 00 00 00  	v.nop
   209b4: 00 00 00 00 00 00    	m.nop
   209ba: 65 ce 9d 01  	l.andix r25, r25, 0x1d
   209be: 01 00        	c.nop
   209c0: 00 00 00 00  	v.nop
   209c4: 00 00 00 00 00 00    	m.nop
   209ca: 25 3e 8a 01  	l.andix r24, r7, 0xa
   209ce: 01 00        	c.nop
   209d0: 00 00 00 00  	v.nop
   209d4: 00 00 00 00 00 00    	m.nop
   209da: ef 3f 7f 1f  	l.ldi16.u r7, 0xffff
   209de: 01 00        	c.nop
   209e0: 00 00 00 00  	v.nop
   209e4: 00 00 00 00 00 00    	m.nop
   209ea: 25 63 87 01  	l.andix r12, r12, 0x7
   209ee: 01 00        	c.nop
   209f0: 00 00 00 00  	v.nop
   209f4: 00 00 00 00 00 00    	m.nop
   209fa: a5 d6 8a 01  	l.andix r26, r26, 0xa
   209fe: 01 00        	c.nop
   20a00: 00 00 00 00  	v.nop
   20a04: 00 00 00 00 00 00    	m.nop
   20a0a: a5 94 8a 01  	l.andix r18, r18, 0xa
   20a0e: 01 00        	c.nop
   20a10: 00 00 00 00  	v.nop
   20a14: 00 00 00 00 00 00    	m.nop
   20a1a: 65 8c 8a 01  	l.andix r17, r17, 0xa
   20a1e: 01 00        	c.nop
   20a20: 00 00 00 00  	v.nop
   20a24: 00 00 00 00 00 00    	m.nop
   20a2a: a5 aa 9d 01  	l.andix r10, r21, 0x1d
   20a2e: 01 00        	c.nop
   20a30: 00 00 00 00  	v.nop
   20a34: 00 00 00 00 00 00    	m.nop
   20a3a: f1 31 58 08  	l.ld.w r7, [r6, 0x98]
   20a3e: 01 00        	c.nop
   20a40: 00 00 00 00  	v.nop
   20a44: 00 00 00 00 00 00    	m.nop
   20a4a: a5 d2 0a 02  	l.orix r10, r26, 0xa
   20a4e: 01 00        	c.nop
   20a50: 00 00 00 00  	v.nop
   20a54: 00 00 00 00 00 00    	m.nop
   20a5a: a3 7e 1c 18  	<unknown>
   20a5e: 01 00        	c.nop
   20a60: 00 00 00 00  	v.nop
   20a64: 00 00 00 00 00 00    	m.nop
   20a6a: 63 9d 15 18  	<unknown>
   20a6e: 01 00        	c.nop
   20a70: 00 00 00 00  	v.nop
   20a74: 00 00 00 00 00 00    	m.nop
   20a7a: 65 d5 15 02  	l.orix r21, r26, 0x15
   20a7e: 01 00        	c.nop
   20a80: 00 00 00 00  	v.nop
   20a84: 00 00 00 00 00 00    	m.nop
   20a8a: a3 b6 0d 18  	<unknown>
   20a8e: 01 00        	c.nop
   20a90: 00 00 00 00  	v.nop
   20a94: 00 00 00 00 00 00    	m.nop
   20a9a: 65 ad 1a 02  	l.orix r21, r21, 0x1a
   20a9e: 01 00        	c.nop
   20aa0: 00 00 00 00  	v.nop
   20aa4: 00 00 00 00 00 00    	m.nop
   20aaa: a5 2e 9d 01  	l.andix r26, r5, 0x1d
   20aae: ee 92        	<unknown>
   20ab0: 00 00 00 00  	v.nop
   20ab4: 00 00 00 00 00 00    	m.nop
   20aba: 65 ad 05 02  	l.orix r21, r21, 0x5
   20abe: f1 42        	<unknown>
   20ac0: 00 00 00 00  	v.nop
   20ac4: 00 00 00 00 00 00    	m.nop
   20aca: e3 3a 05 10  	<unknown>
   20ace: 01 00        	c.nop
   20ad0: 00 00 00 00  	v.nop
   20ad4: 00 00 00 00 00 00    	m.nop
   20ada: 71 11 5c 04  	l.ld.w r5, [sp, 0x5c]
   20ade: 01 00        	c.nop
   20ae0: 00 00 00 00  	v.nop
   20ae4: 00 00 00 00 00 00    	m.nop
   20aea: a5 b5 9d 01  	l.andix r22, r22, 0x1d
   20aee: 01 00        	c.nop
   20af0: 00 00 00 00  	v.nop
   20af4: 00 00 00 00 00 00    	m.nop
   20afa: 25 e7 9d 01  	l.andix r28, r28, 0x1d
   20afe: 01 00        	c.nop
   20b00: 00 00 00 00  	v.nop
   20b04: 00 00 00 00 00 00    	m.nop
   20b0a: 2f 40 80 30  	l.ldi_hi.m1 r8, 0x8000
   20b0e: 01 00        	c.nop
   20b10: 00 00 00 00  	v.nop
   20b14: 00 00 00 00 00 00    	m.nop
   20b1a: 31 58 c5 00  	l.st.w r5, [r11]
   20b1e: 01 00        	c.nop
   20b20: 00 00 00 00  	v.nop
   20b24: 00 00 00 00 00 00    	m.nop
   20b2a: 71 11 54 12  	l.ld.w r5, [sp, 0x134]
   20b2e: 01 00        	c.nop
   20b30: 00 00 00 00  	v.nop
   20b34: 00 00 00 00 00 00    	m.nop
   20b3a: 31 5f c5 fe  	l.st.w r5, [r11, -0x4]
   20b3e: 01 00        	c.nop
   20b40: 00 00 00 00  	v.nop
   20b44: 00 00 00 00 00 00    	m.nop
   20b4a: 71 11 40 0c  	l.ld.w r5, [sp, 0xc0]
   20b4e: 01 00        	c.nop
   20b50: 00 00 00 00  	v.nop
   20b54: 00 00 00 00 00 00    	m.nop
   20b5a: 31 5a c5 00  	l.st.w r5, [r11, 0x8]
   20b5e: 01 00        	c.nop
   20b60: 00 00 00 00  	v.nop
   20b64: 00 00 00 00 00 00    	m.nop
   20b6a: 65 b1 12 02  	l.orix r5, r22, 0x12
   20b6e: 01 00        	c.nop
   20b70: 00 00 00 00  	v.nop
   20b74: 00 00 00 00 00 00    	m.nop
   20b7a: 31 59 cc 00  	l.st.w r12, [r11, 0x4]
   20b7e: 01 00        	c.nop
   20b80: 00 00 00 00  	v.nop
   20b84: 00 00 00 00 00 00    	m.nop
   20b8a: 31 5a d7 0a  	l.st.w r23, [r11, 0xa8]
   20b8e: 01 00        	c.nop
   20b90: 00 00 00 00  	v.nop
   20b94: 00 00 00 00 00 00    	m.nop
   20b9a: 31 5b ce 0a  	l.st.w r14, [r11, 0xac]
   20b9e: 01 00        	c.nop
   20ba0: 00 00 00 00  	v.nop
   20ba4: 00 00 00 00 00 00    	m.nop
   20baa: 31 5c d4 0a  	l.st.w r20, [r11, 0xb0]
   20bae: 01 00        	c.nop
   20bb0: 00 00 00 00  	v.nop
   20bb4: 00 00 00 00 00 00    	m.nop
   20bba: 31 5d d0 0a  	l.st.w r16, [r11, 0xb4]
   20bbe: 01 00        	c.nop
   20bc0: 00 00 00 00  	v.nop
   20bc4: 00 00 00 00 00 00    	m.nop
   20bca: 31 5c d9 00  	l.st.w r25, [r11, 0x10]
   20bce: 01 00        	c.nop
   20bd0: 00 00 00 00  	v.nop
   20bd4: 00 00 00 00 00 00    	m.nop
   20bda: 31 59 ca 0c  	l.st.w r10, [r11, 0xc4]
   20bde: 01 00        	c.nop
   20be0: 00 00 00 00  	v.nop
   20be4: 00 00 00 00 00 00    	m.nop
   20bea: 31 5a c5 0c  	l.st.w r5, [r11, 0xc8]
   20bee: 01 00        	c.nop
   20bf0: 00 00 00 00  	v.nop
   20bf4: 00 00 00 00 00 00    	m.nop
   20bfa: a5 c2 1c 02  	l.orix r10, r24, 0x1c
   20bfe: 01 00        	c.nop
   20c00: 00 00 00 00  	v.nop
   20c04: 00 00 00 00 00 00    	m.nop
   20c0a: 65 d1 11 02  	l.orix r5, r26, 0x11
   20c0e: 01 00        	c.nop
   20c10: 00 00 00 00  	v.nop
   20c14: 00 00 00 00 00 00    	m.nop
   20c1a: 31 5b ca 0c  	l.st.w r10, [r11, 0xcc]
   20c1e: 01 00        	c.nop
   20c20: 00 00 00 00  	v.nop
   20c24: 00 00 00 00 00 00    	m.nop
   20c2a: 31 59 c5 06  	l.st.w r5, [r11, 0x64]
   20c2e: 01 00        	c.nop
   20c30: 00 00 00 00  	v.nop
   20c34: 00 00 00 00 00 00    	m.nop
   20c3a: a5 9a 9d 01  	l.andix r10, r19, 0x1d
   20c3e: 01 00        	c.nop
   20c40: 00 00 00 00  	v.nop
   20c44: 00 00 00 00 00 00    	m.nop
   20c4a: 65 69 9d 01  	l.andix r5, r13, 0x1d
   20c4e: 01 00        	c.nop
   20c50: 00 00 00 00  	v.nop
   20c54: 00 00 00 00 00 00    	m.nop
   20c5a: 31 59 ca 0a  	l.st.w r10, [r11, 0xa4]
   20c5e: 01 00        	c.nop
   20c60: 00 00 00 00  	v.nop
   20c64: 00 00 00 00 00 00    	m.nop
   20c6a: 31 5e c5 0a  	l.st.w r5, [r11, 0xb8]
   20c6e: 01 00        	c.nop
   20c70: 00 00 00 00  	v.nop
   20c74: 00 00 00 00 00 00    	m.nop
   20c7a: a5 7a 9d 01  	l.andix r10, r15, 0x1d
   20c7e: 01 00        	c.nop
   20c80: 00 00 00 00  	v.nop
   20c84: 00 00 00 00 00 00    	m.nop
   20c8a: 65 d9 9d 01  	l.andix r5, r27, 0x1d
   20c8e: 01 00        	c.nop
   20c90: 00 00 00 00  	v.nop
   20c94: 00 00 00 00 00 00    	m.nop
   20c9a: 31 5f ca 0a  	l.st.w r10, [r11, 0xbc]
   20c9e: 01 00        	c.nop
   20ca0: 00 00 00 00  	v.nop
   20ca4: 00 00 00 00 00 00    	m.nop
   20caa: 31 58 c5 0c  	l.st.w r5, [r11, 0xc0]
   20cae: 81 42        	<unknown>
   20cb0: 00 00 00 00  	v.nop
   20cb4: 00 00 00 00 00 00    	m.nop
   20cba: 2f 50 88 20  	l.ldi_hi.m1 r10, 0x100
   20cbe: 01 00        	c.nop
   20cc0: 00 00 00 00  	v.nop
   20cc4: 00 00 00 00 00 00    	m.nop
   20cca: 31 5c c0 0c  	l.st.w r0, [r11, 0xd0]
   20cce: 01 00        	c.nop
   20cd0: 00 00 00 00  	v.nop
   20cd4: 00 00 00 00 00 00    	m.nop
   20cda: bb a8 0a c0  	l.blt.u r21, r10, 0x2 <0x20cf0>
   20cde: 01 00        	c.nop
   20ce0: 00 00 00 00  	v.nop
   20ce4: 00 00 00 00 00 00    	m.nop
   20cea: 00 00        	s.nop16
   20cec: 00 00        	s.nop16
   20cee: 16 84        	<unknown>
   20cf0: 00 00 00 00  	v.nop
   20cf4: 00 00 00 00 00 00    	m.nop
   20cfa: 71 11 44 0c  	l.ld.w r5, [sp, 0xc4]
   20cfe: 85 45        	<unknown>
   20d00: 00 00 00 00  	v.nop
   20d04: 00 00 00 00 00 00    	m.nop
   20d0a: 2f 50 18 21  	l.ldi16.s r10, 0x300
   20d0e: 01 46        	<unknown>
   20d10: 00 00 00 00  	v.nop
   20d14: 00 00 00 00 00 00    	m.nop
   20d1a: 00 00        	s.nop16
   20d1c: 00 00        	s.nop16
   20d1e: 09 47        	<unknown>
   20d20: 00 00 00 00  	v.nop
   20d24: 00 00 00 00 00 00    	m.nop
   20d2a: 00 00        	s.nop16
   20d2c: 00 00        	s.nop16
   20d2e: 9e 02        	<unknown>
   20d30: 00 00 00 00  	v.nop
   20d34: 00 00 00 00 00 00    	m.nop
   20d3a: 65 29 8a 01  	l.andix r5, r5, 0xa
   20d3e: 01 00        	c.nop
   20d40: 00 00 00 00  	v.nop
   20d44: 00 00 00 00 00 00    	m.nop
   20d4a: 65 29 08 02  	l.orix r5, r5, 0x8
   20d4e: 01 00        	c.nop
   20d50: 00 00 00 00  	v.nop
   20d54: 00 00 00 00 00 00    	m.nop
   20d5a: 2f 42 10 21  	l.ldi16.s r8, 0x208
   20d5e: 01 00        	c.nop
   20d60: 00 00 00 00  	v.nop
   20d64: 00 00 00 00 00 00    	m.nop
   20d6a: 2f 40 81 00  	l.ldi_hi r8, 0x20
   20d6e: 01 00        	c.nop
   20d70: 00 00 00 00  	v.nop
   20d74: 00 00 00 00 00 00    	m.nop
   20d7a: e1 43 0c 2e  	l.addi.s r15, r8, -0x1f4
   20d7e: 51 44        	<unknown>
   20d80: 00 00 00 00  	v.nop
   20d84: 00 00 00 00 00 00    	m.nop
   20d8a: 65 29 0f 02  	l.orix r5, r5, 0xf
   20d8e: a2 93        	<unknown>
   20d90: 00 00 00 00  	v.nop
   20d94: 00 00 00 00 00 00    	m.nop
   20d9a: 31 38 c5 00  	l.st.w r5, [r7]
   20d9e: 01 00        	c.nop
   20da0: 00 00 00 00  	v.nop
   20da4: 00 00 00 00 00 00    	m.nop
   20daa: 71 11 58 12  	l.ld.w r5, [sp, 0x138]
   20dae: 01 00        	c.nop
   20db0: 00 00 00 00  	v.nop
   20db4: 00 00 00 00 00 00    	m.nop
   20dba: f1 11 48 0c  	l.ld.w r7, [sp, 0xc8]
   20dbe: 01 00        	c.nop
   20dc0: 00 00 00 00  	v.nop
   20dc4: 00 00 00 00 00 00    	m.nop
   20dca: 71 4b 44 00  	l.ld.w r13, [r9, 0x4]
   20dce: 01 00        	c.nop
   20dd0: 00 00 00 00  	v.nop
   20dd4: 00 00 00 00 00 00    	m.nop
   20dda: 71 4c 40 00  	l.ld.w r17, [r9]
   20dde: 01 00        	c.nop
   20de0: 00 00 00 00  	v.nop
   20de4: 00 00 00 00 00 00    	m.nop
   20dea: 31 4a 4c 02  	l.ld.w r8, [r9, 0x2c]
   20dee: 01 00        	c.nop
   20df0: 00 00 00 00  	v.nop
   20df4: 00 00 00 00 00 00    	m.nop
   20dfa: ef 57 07 21  	l.ldi16.s r10, 0xff
   20dfe: 01 00        	c.nop
   20e00: 00 00 00 00  	v.nop
   20e04: 00 00 00 00 00 00    	m.nop
   20e0a: e3 29 07 18  	<unknown>
   20e0e: 01 00        	c.nop
   20e10: 00 00 00 00  	v.nop
   20e14: 00 00 00 00 00 00    	m.nop
   20e1a: 71 39 08 00  	l.ld.b r5, [r7, 0x8]
   20e1e: 01 00        	c.nop
   20e20: 00 00 00 00  	v.nop
   20e24: 00 00 00 00 00 00    	m.nop
   20e2a: 65 29 8a 01  	l.andix r5, r5, 0xa
   20e2e: 01 00        	c.nop
   20e30: 00 00 00 00  	v.nop
   20e34: 00 00 00 00 00 00    	m.nop
   20e3a: 7b 29 00 00  	<unknown>
   20e3e: 01 00        	c.nop
   20e40: 00 00 00 00  	v.nop
   20e44: 00 00 00 00 00 00    	m.nop
   20e4a: bb 41 0e c0  	l.blt.u r8, r14, 0x6 <0x20ea0>
   20e4e: 01 00        	c.nop
   20e50: 00 00 00 00  	v.nop
   20e54: 00 00 00 00 00 00    	m.nop
   20e5a: fb 41 0e 40  	<unknown>
   20e5e: 01 00        	c.nop
   20e60: 00 00 00 00  	v.nop
   20e64: 00 00 00 00 00 00    	m.nop
   20e6a: 00 00        	s.nop16
   20e6c: 00 00        	s.nop16
   20e6e: 85 42        	<unknown>
   20e70: 00 00 00 00  	v.nop
   20e74: 00 00 00 00 00 00    	m.nop
   20e7a: 2b 30 80 01  	l.jal r0, 0x6 <0x20ed0>
   20e7e: 01 00        	c.nop
   20e80: 00 00 00 00  	v.nop
   20e84: 00 00 00 00 00 00    	m.nop
   20e8a: 00 00        	s.nop16
   20e8c: 00 00        	s.nop16
   20e8e: 85 42        	<unknown>
   20e90: 00 00 00 00  	v.nop
   20e94: 00 00 00 00 00 00    	m.nop
   20e9a: 3b 41 05 00  	<unknown>
   20e9e: 01 00        	c.nop
   20ea0: 00 00 00 00  	v.nop
   20ea4: 00 00 00 00 00 00    	m.nop
   20eaa: 00 00        	s.nop16
   20eac: 00 00        	s.nop16
   20eae: 81 42        	<unknown>
   20eb0: 00 00 00 00  	v.nop
   20eb4: 00 00 00 00 00 00    	m.nop
   20eba: 2b 10 80 01  	l.jal r0, 0x2 <0x20ed0>
   20ebe: 01 00        	c.nop
   20ec0: 00 00 00 00  	v.nop
   20ec4: 00 00 00 00 00 00    	m.nop
   20eca: 00 00        	s.nop16
   20ecc: 00 00        	s.nop16
   20ece: 8d 42        	<unknown>
   20ed0: 00 00 00 00  	v.nop
   20ed4: 00 00 00 00 00 00    	m.nop
   20eda: 31 4a 48 00  	l.ld.w r8, [r9, 0x8]
   20ede: 51 45        	<unknown>
   20ee0: 00 00 00 00  	v.nop
   20ee4: 00 00 00 00 00 00    	m.nop
   20eea: 31 3f ca 0c  	l.st.w r10, [r7, 0xdc]
   20eee: 01 00        	c.nop
   20ef0: 00 00 00 00  	v.nop
   20ef4: 00 00 00 00 00 00    	m.nop
   20efa: b1 12 50 04  	l.ld.w r10, [sp, 0x50]
   20efe: 01 00        	c.nop
   20f00: 00 00 00 00  	v.nop
   20f04: 00 00 00 00 00 00    	m.nop
   20f0a: 31 4c 48 02  	l.ld.w r16, [r9, 0x28]
   20f0e: 01 00        	c.nop
   20f10: 00 00 00 00  	v.nop
   20f14: 00 00 00 00 00 00    	m.nop
   20f1a: 31 39 c5 0e  	l.st.w r5, [r7, 0xe4]
   20f1e: 95 42        	<unknown>
   20f20: 00 00 00 00  	v.nop
   20f24: 00 00 00 00 00 00    	m.nop
   20f2a: 63 29 88 18  	<unknown>
   20f2e: 01 00        	c.nop
   20f30: 00 00 00 00  	v.nop
   20f34: 00 00 00 00 00 00    	m.nop
   20f3a: 31 3d cd 0e  	l.st.w r13, [r7, 0xf4]
   20f3e: 01 00        	c.nop
   20f40: 00 00 00 00  	v.nop
   20f44: 00 00 00 00 00 00    	m.nop
   20f4a: 31 3f ca 0e  	l.st.w r10, [r7, 0xfc]
   20f4e: 01 00        	c.nop
   20f50: 00 00 00 00  	v.nop
   20f54: 00 00 00 00 00 00    	m.nop
   20f5a: b1 12 58 04  	l.ld.w r10, [sp, 0x58]
   20f5e: 01 00        	c.nop
   20f60: 00 00 00 00  	v.nop
   20f64: 00 00 00 00 00 00    	m.nop
   20f6a: 31 3c d1 0e  	l.st.w r17, [r7, 0xf0]
   20f6e: 01 00        	c.nop
   20f70: 00 00 00 00  	v.nop
   20f74: 00 00 00 00 00 00    	m.nop
   20f7a: 31 3f cc 1c  	l.st.w r12, [r7, 0x1dc]
   20f7e: 01 00        	c.nop
   20f80: 00 00 00 00  	v.nop
   20f84: 00 00 00 00 00 00    	m.nop
   20f8a: 31 38 d0 0e  	l.st.w r16, [r7, 0xe0]
   20f8e: 01 00        	c.nop
   20f90: 00 00 00 00  	v.nop
   20f94: 00 00 00 00 00 00    	m.nop
   20f9a: 31 38 c8 12  	l.st.w r8, [r7, 0x120]
   20f9e: 01 00        	c.nop
   20fa0: 00 00 00 00  	v.nop
   20fa4: 00 00 00 00 00 00    	m.nop
   20faa: 31 3e ca 0e  	l.st.w r10, [r7, 0xf8]
   20fae: 01 00        	c.nop
   20fb0: 00 00 00 00  	v.nop
   20fb4: 00 00 00 00 00 00    	m.nop
   20fba: 31 11 cf 0c  	l.st.w r15, [sp, 0xc4]
   20fbe: 01 00        	c.nop
   20fc0: 00 00 00 00  	v.nop
   20fc4: 00 00 00 00 00 00    	m.nop
   20fca: 31 12 d0 0c  	l.st.w r16, [sp, 0xc8]
   20fce: 01 00        	c.nop
   20fd0: 00 00 00 00  	v.nop
   20fd4: 00 00 00 00 00 00    	m.nop
   20fda: 31 10 d1 0c  	l.st.w r17, [sp, 0xc0]
   20fde: 01 00        	c.nop
   20fe0: 00 00 00 00  	v.nop
   20fe4: 00 00 00 00 00 00    	m.nop
   20fea: bb 5b a5 80  	l.bge.s r11, r5, 0x2e <0x212c0>
   20fee: 01 00        	c.nop
   20ff0: 00 00 00 00  	v.nop
   20ff4: 00 00 00 00 00 00    	m.nop
   20ffa: fb 2a 4e 00  	<unknown>
   20ffe: 01 00        	c.nop
   21000: 00 00 00 00  	v.nop
   21004: 00 00 00 00 00 00    	m.nop
   2100a: 00 00        	s.nop16
   2100c: 00 00        	s.nop16
   2100e: 0d 44        	<unknown>
   21010: 00 00 00 00  	v.nop
   21014: 00 00 00 00 00 00    	m.nop
   2101a: 3b 2a 68 00  	<unknown>
   2101e: 01 00        	c.nop
   21020: 00 00 00 00  	v.nop
   21024: 00 00 00 00 00 00    	m.nop
   2102a: 00 00        	s.nop16
   2102c: 00 00        	s.nop16
   2102e: 11 44        	<unknown>
   21030: 00 00 00 00  	v.nop
   21034: 00 00 00 00 00 00    	m.nop
   2103a: 3b 2a 28 41  	<unknown>
   2103e: 01 00        	c.nop
   21040: 00 00 00 00  	v.nop
   21044: 00 00 00 00 00 00    	m.nop
   2104a: 71 11 40 02  	l.ld.w r5, [sp, 0x20]
   2104e: 85 47        	<unknown>
   21050: 00 00 00 00  	v.nop
   21054: 00 00 00 00 00 00    	m.nop
   2105a: 71 4a 4c 00  	l.ld.w r9, [r9, 0xc]
   2105e: 81 4d        	<unknown>
   21060: 00 00 00 00  	v.nop
   21064: 00 00 00 00 00 00    	m.nop
   2106a: 31 17 40 06  	l.ld.w r28, [sp, 0x60]
   2106e: 01 00        	c.nop
   21070: 00 00 00 00  	v.nop
   21074: 00 00 00 00 00 00    	m.nop
   2107a: 31 12 44 04  	l.ld.w r8, [sp, 0x44]
   2107e: 01 00        	c.nop
   21080: 00 00 00 00  	v.nop
   21084: 00 00 00 00 00 00    	m.nop
   2108a: a3 26 9f 03  	l.srli r26, r4, 0x1f
   2108e: 01 00        	c.nop
   21090: 00 00 00 00  	v.nop
   21094: 00 00 00 00 00 00    	m.nop
   2109a: 31 3d c5 12  	l.st.w r5, [r7, 0x134]
   2109e: 01 00        	c.nop
   210a0: 00 00 00 00  	v.nop
   210a4: 00 00 00 00 00 00    	m.nop
   210aa: 2f 28 78 3f  	l.ldi16.s r5, -0x100
   210ae: 01 00        	c.nop
   210b0: 00 00 00 00  	v.nop
   210b4: 00 00 00 00 00 00    	m.nop
   210ba: 31 39 cf 12  	l.st.w r15, [r7, 0x124]
   210be: 01 00        	c.nop
   210c0: 00 00 00 00  	v.nop
   210c4: 00 00 00 00 00 00    	m.nop
   210ca: 31 3e cf 12  	l.st.w r15, [r7, 0x138]
   210ce: 01 00        	c.nop
   210d0: 00 00 00 00  	v.nop
   210d4: 00 00 00 00 00 00    	m.nop
   210da: 31 3b c0 14  	l.st.w r0, [r7, 0x14c]
   210de: 01 00        	c.nop
   210e0: 00 00 00 00  	v.nop
   210e4: 00 00 00 00 00 00    	m.nop
   210ea: 31 3d c0 16  	l.st.w r0, [r7, 0x174]
   210ee: 01 00        	c.nop
   210f0: 00 00 00 00  	v.nop
   210f4: 00 00 00 00 00 00    	m.nop
   210fa: 31 3a cf 12  	l.st.w r15, [r7, 0x128]
   210fe: 01 00        	c.nop
   21100: 00 00 00 00  	v.nop
   21104: 00 00 00 00 00 00    	m.nop
   2110a: 31 3f cf 12  	l.st.w r15, [r7, 0x13c]
   2110e: 01 00        	c.nop
   21110: 00 00 00 00  	v.nop
   21114: 00 00 00 00 00 00    	m.nop
   2111a: 31 3c c0 14  	l.st.w r0, [r7, 0x150]
   2111e: 01 00        	c.nop
   21120: 00 00 00 00  	v.nop
   21124: 00 00 00 00 00 00    	m.nop
   2112a: 31 3e c0 16  	l.st.w r0, [r7, 0x178]
   2112e: 01 00        	c.nop
   21130: 00 00 00 00  	v.nop
   21134: 00 00 00 00 00 00    	m.nop
   2113a: 31 3b cf 12  	l.st.w r15, [r7, 0x12c]
   2113e: 01 00        	c.nop
   21140: 00 00 00 00  	v.nop
   21144: 00 00 00 00 00 00    	m.nop
   2114a: 31 38 cf 14  	l.st.w r15, [r7, 0x140]
   2114e: 01 00        	c.nop
   21150: 00 00 00 00  	v.nop
   21154: 00 00 00 00 00 00    	m.nop
   2115a: 31 3d c0 14  	l.st.w r0, [r7, 0x154]
   2115e: 01 00        	c.nop
   21160: 00 00 00 00  	v.nop
   21164: 00 00 00 00 00 00    	m.nop
   2116a: 31 3f c0 16  	l.st.w r0, [r7, 0x17c]
   2116e: 01 00        	c.nop
   21170: 00 00 00 00  	v.nop
   21174: 00 00 00 00 00 00    	m.nop
   2117a: 31 3c cf 12  	l.st.w r15, [r7, 0x130]
   2117e: 01 00        	c.nop
   21180: 00 00 00 00  	v.nop
   21184: 00 00 00 00 00 00    	m.nop
   2118a: 31 39 cf 14  	l.st.w r15, [r7, 0x144]
   2118e: 01 00        	c.nop
   21190: 00 00 00 00  	v.nop
   21194: 00 00 00 00 00 00    	m.nop
   2119a: 31 3e c0 14  	l.st.w r0, [r7, 0x158]
   2119e: 01 00        	c.nop
   211a0: 00 00 00 00  	v.nop
   211a4: 00 00 00 00 00 00    	m.nop
   211aa: 31 38 c0 18  	l.st.w r0, [r7, 0x180]
   211ae: 01 00        	c.nop
   211b0: 00 00 00 00  	v.nop
   211b4: 00 00 00 00 00 00    	m.nop
   211ba: 31 3a c9 14  	l.st.w r9, [r7, 0x148]
   211be: 01 00        	c.nop
   211c0: 00 00 00 00  	v.nop
   211c4: 00 00 00 00 00 00    	m.nop
   211ca: 31 3f c4 14  	l.st.w r4, [r7, 0x15c]
   211ce: 01 00        	c.nop
   211d0: 00 00 00 00  	v.nop
   211d4: 00 00 00 00 00 00    	m.nop
   211da: 31 39 da 18  	l.st.w r26, [r7, 0x184]
   211de: 01 00        	c.nop
   211e0: 00 00 00 00  	v.nop
   211e4: 00 00 00 00 00 00    	m.nop
   211ea: bb 40 85 c0  	l.blt.s r8, r5, 0x2 <0x21200>
   211ee: 01 00        	c.nop
   211f0: 00 00 00 00  	v.nop
   211f4: 00 00 00 00 00 00    	m.nop
   211fa: 00 00        	s.nop16
   211fc: 00 00        	s.nop16
   211fe: 12 8e        	<unknown>
   21200: 00 00 00 00  	v.nop
   21204: 00 00 00 00 00 00    	m.nop
   2120a: ef ef 7f 1f  	l.ldi16.u r29, 0xffff
   2120e: 3e 87        	<unknown>
   21210: 00 00 00 00  	v.nop
   21214: 00 00 00 00 00 00    	m.nop
   2121a: 71 14 4c 08  	l.ld.w r17, [sp, 0x8c]
   2121e: ee 82        	<unknown>
   21220: 00 00 00 00  	v.nop
   21224: 00 00 00 00 00 00    	m.nop
   2122a: ef ef 87 00  	l.ldi_hi r29, 0xff
   2122e: 3e 8a        	<unknown>
   21230: 00 00 00 00  	v.nop
   21234: 00 00 00 00 00 00    	m.nop
   2123a: 00 00        	s.nop16
   2123c: 00 00        	s.nop16
   2123e: 3e 88        	<unknown>
   21240: 00 00 00 00  	v.nop
   21244: 00 00 00 00 00 00    	m.nop
   2124a: 00 00        	s.nop16
   2124c: 00 00        	s.nop16
   2124e: 6e 85        	<unknown>
   21250: 00 00 00 00  	v.nop
   21254: 00 00 00 00 00 00    	m.nop
   2125a: 00 00        	s.nop16
   2125c: 00 00        	s.nop16
   2125e: be 8c        	<unknown>
   21260: 00 00 00 00  	v.nop
   21264: 00 00 00 00 00 00    	m.nop
   2126a: 00 00        	s.nop16
   2126c: 00 00        	s.nop16
   2126e: be 89        	<unknown>
   21270: 00 00 00 00  	v.nop
   21274: 00 00 00 00 00 00    	m.nop
   2127a: 00 00        	s.nop16
   2127c: 00 00        	s.nop16
   2127e: 6e 84        	<unknown>
   21280: 00 00 00 00  	v.nop
   21284: 00 00 00 00 00 00    	m.nop
   2128a: 00 00        	s.nop16
   2128c: 00 00        	s.nop16
   2128e: be 8a        	<unknown>
   21290: 00 00 00 00  	v.nop
   21294: 00 00 00 00 00 00    	m.nop
   2129a: 00 00        	s.nop16
   2129c: 00 00        	s.nop16
   2129e: 6e 89        	<unknown>
   212a0: 00 00 00 00  	v.nop
   212a4: 00 00 00 00 00 00    	m.nop
   212aa: 00 00        	s.nop16
   212ac: 00 00        	s.nop16
   212ae: 6e 8b        	<unknown>
   212b0: 00 00 00 00  	v.nop
   212b4: 00 00 00 00 00 00    	m.nop
   212ba: 2b 00 8a 01  	l.jal r0, 0xa0 <0x21cb0>
   212be: 01 00        	c.nop
   212c0: 00 00 00 00  	v.nop
   212c4: 00 00 00 00 00 00    	m.nop
   212ca: bb 2f 40 00  	<unknown>
   212ce: 01 00        	c.nop
   212d0: 00 00 00 00  	v.nop
   212d4: 00 00 00 00 00 00    	m.nop
   212da: bb 2f 6b 40  	<unknown>
   212de: 01 00        	c.nop
   212e0: 00 00 00 00  	v.nop
   212e4: 00 00 00 00 00 00    	m.nop
   212ea: 31 13 54 02  	l.ld.w r12, [sp, 0x34]
   212ee: 01 00        	c.nop
   212f0: 00 00 00 00  	v.nop
   212f4: 00 00 00 00 00 00    	m.nop
   212fa: b1 15 50 02  	l.ld.w r22, [sp, 0x30]
   212fe: 01 00        	c.nop
   21300: 00 00 00 00  	v.nop
   21304: 00 00 00 00 00 00    	m.nop
   2130a: f1 15 58 02  	l.ld.w r23, [sp, 0x38]
   2130e: 01 00        	c.nop
   21310: 00 00 00 00  	v.nop
   21314: 00 00 00 00 00 00    	m.nop
   2131a: 31 16 40 04  	l.ld.w r24, [sp, 0x40]
   2131e: 01 00        	c.nop
   21320: 00 00 00 00  	v.nop
   21324: 00 00 00 00 00 00    	m.nop
   2132a: 71 15 48 02  	l.ld.w r21, [sp, 0x28]
   2132e: 01 00        	c.nop
   21330: 00 00 00 00  	v.nop
   21334: 00 00 00 00 00 00    	m.nop
   2133a: 71 14 40 02  	l.ld.w r17, [sp, 0x20]
   2133e: 32 8d        	<unknown>
   21340: 00 00 00 00  	v.nop
   21344: 00 00 00 00 00 00    	m.nop
   2134a: f1 4c 4c 00  	l.ld.w r19, [r9, 0xc]
   2134e: da 87        	<unknown>
   21350: 00 00 00 00  	v.nop
   21354: 00 00 00 00 00 00    	m.nop
   2135a: 31 4c 50 00  	l.ld.w r16, [r9, 0x10]
   2135e: 5e 85        	<unknown>
   21360: 00 00 00 00  	v.nop
   21364: 00 00 00 00 00 00    	m.nop
   2136a: b1 4b 54 00  	l.ld.w r14, [r9, 0x14]
   2136e: e2 82        	<unknown>
   21370: 00 00 00 00  	v.nop
   21374: 00 00 00 00 00 00    	m.nop
   2137a: 23 22 9f 03  	l.srli r8, r4, 0x1f
   2137e: 56 8a        	<unknown>
   21380: 00 00 00 00  	v.nop
   21384: 00 00 00 00 00 00    	m.nop
   2138a: 31 39 cb 12  	l.st.w r11, [r7, 0x124]
   2138e: c6 8c        	<unknown>
   21390: 00 00 00 00  	v.nop
   21394: 00 00 00 00 00 00    	m.nop
   2139a: 31 3e cb 12  	l.st.w r11, [r7, 0x138]
   2139e: 81 45        	<unknown>
   213a0: 00 00 00 00  	v.nop
   213a4: 00 00 00 00 00 00    	m.nop
   213aa: 31 3e d7 14  	l.st.w r23, [r7, 0x158]
   213ae: ea 8b        	<unknown>
   213b0: 00 00 00 00  	v.nop
   213b4: 00 00 00 00 00 00    	m.nop
   213ba: b1 14 40 10  	l.ld.w r18, [sp, 0x100]
   213be: e1 04        	<unknown>
   213c0: 00 00 00 00  	v.nop
   213c4: 00 00 00 00 00 00    	m.nop
   213ca: 31 3b cb 14  	l.st.w r11, [r7, 0x14c]
   213ce: 01 00        	c.nop
   213d0: 00 00 00 00  	v.nop
   213d4: 00 00 00 00 00 00    	m.nop
   213da: 31 3d cb 16  	l.st.w r11, [r7, 0x174]
   213de: 01 00        	c.nop
   213e0: 00 00 00 00  	v.nop
   213e4: 00 00 00 00 00 00    	m.nop
   213ea: 31 3a d1 12  	l.st.w r17, [r7, 0x128]
   213ee: 01 00        	c.nop
   213f0: 00 00 00 00  	v.nop
   213f4: 00 00 00 00 00 00    	m.nop
   213fa: 31 3f d3 12  	l.st.w r19, [r7, 0x13c]
   213fe: 01 00        	c.nop
   21400: 00 00 00 00  	v.nop
   21404: 00 00 00 00 00 00    	m.nop
   2140a: 31 3c c4 14  	l.st.w r4, [r7, 0x150]
   2140e: 01 00        	c.nop
   21410: 00 00 00 00  	v.nop
   21414: 00 00 00 00 00 00    	m.nop
   2141a: 31 3e c8 16  	l.st.w r8, [r7, 0x178]
   2141e: 01 00        	c.nop
   21420: 00 00 00 00  	v.nop
   21424: 00 00 00 00 00 00    	m.nop
   2142a: 31 3b d5 12  	l.st.w r21, [r7, 0x12c]
   2142e: 01 00        	c.nop
   21430: 00 00 00 00  	v.nop
   21434: 00 00 00 00 00 00    	m.nop
   2143a: 31 38 d0 14  	l.st.w r16, [r7, 0x140]
   2143e: 01 00        	c.nop
   21440: 00 00 00 00  	v.nop
   21444: 00 00 00 00 00 00    	m.nop
   2144a: 31 3d cb 14  	l.st.w r11, [r7, 0x154]
   2144e: 01 00        	c.nop
   21450: 00 00 00 00  	v.nop
   21454: 00 00 00 00 00 00    	m.nop
   2145a: 31 3f cb 16  	l.st.w r11, [r7, 0x17c]
   2145e: 01 00        	c.nop
   21460: 00 00 00 00  	v.nop
   21464: 00 00 00 00 00 00    	m.nop
   2146a: 31 3c d6 12  	l.st.w r22, [r7, 0x130]
   2146e: 01 00        	c.nop
   21470: 00 00 00 00  	v.nop
   21474: 00 00 00 00 00 00    	m.nop
   2147a: 31 39 ce 14  	l.st.w r14, [r7, 0x144]
   2147e: 01 00        	c.nop
   21480: 00 00 00 00  	v.nop
   21484: 00 00 00 00 00 00    	m.nop
   2148a: 31 38 d8 18  	l.st.w r24, [r7, 0x180]
   2148e: 01 00        	c.nop
   21490: 00 00 00 00  	v.nop
   21494: 00 00 00 00 00 00    	m.nop
   2149a: 2b 28 83 01  	l.jal r0, 0x35 <0x217e0>
   2149e: 01 00        	c.nop
   214a0: 00 00 00 00  	v.nop
   214a4: 00 00 00 00 00 00    	m.nop
   214aa: f1 12 40 02  	l.ld.w r11, [sp, 0x20]
   214ae: 85 4c        	<unknown>
   214b0: 00 00 00 00  	v.nop
   214b4: 00 00 00 00 00 00    	m.nop
   214ba: b1 15 50 02  	l.ld.w r22, [sp, 0x30]
   214be: 81 42        	<unknown>
   214c0: 00 00 00 00  	v.nop
   214c4: 00 00 00 00 00 00    	m.nop
   214ca: f1 15 40 04  	l.ld.w r23, [sp, 0x40]
   214ce: e6 89        	<unknown>
   214d0: 00 00 00 00  	v.nop
   214d4: 00 00 00 00 00 00    	m.nop
   214da: 31 13 48 02  	l.ld.w r12, [sp, 0x28]
   214de: 96 88        	<unknown>
   214e0: 00 00 00 00  	v.nop
   214e4: 00 00 00 00 00 00    	m.nop
   214ea: 31 4c 4c 00  	l.ld.w r16, [r9, 0xc]
   214ee: 16 84        	<unknown>
   214f0: 00 00 00 00  	v.nop
   214f4: 00 00 00 00 00 00    	m.nop
   214fa: b1 4b 50 00  	l.ld.w r14, [r9, 0x10]
   214fe: 2e 8a        	<unknown>
   21500: 00 00 00 00  	v.nop
   21504: 00 00 00 00 00 00    	m.nop
   2150a: 71 4a 54 00  	l.ld.w r9, [r9, 0x14]
   2150e: 5a 8c        	<unknown>
   21510: 00 00 00 00  	v.nop
   21514: 00 00 00 00 00 00    	m.nop
   2151a: 31 3b cb 12  	l.st.w r11, [r7, 0x12c]
   2151e: 5e 8d        	<unknown>
   21520: 00 00 00 00  	v.nop
   21524: 00 00 00 00 00 00    	m.nop
   2152a: f1 12 58 02  	l.ld.w r11, [sp, 0x38]
   2152e: b2 87        	<unknown>
   21530: 00 00 00 00  	v.nop
   21534: 00 00 00 00 00 00    	m.nop
   2153a: a3 22 9f 03  	l.srli r10, r4, 0x1f
   2153e: e6 8a        	<unknown>
   21540: 00 00 00 00  	v.nop
   21544: 00 00 00 00 00 00    	m.nop
   2154a: 31 39 d7 18  	l.st.w r23, [r7, 0x184]
   2154e: 16 89        	<unknown>
   21550: 00 00 00 00  	v.nop
   21554: 00 00 00 00 00 00    	m.nop
   2155a: 31 17 48 08  	l.ld.w r28, [sp, 0x88]
   2155e: e2 8b        	<unknown>
   21560: 00 00 00 00  	v.nop
   21564: 00 00 00 00 00 00    	m.nop
   2156a: 31 39 d9 12  	l.st.w r25, [r7, 0x124]
   2156e: 01 00        	c.nop
   21570: 00 00 00 00  	v.nop
   21574: 00 00 00 00 00 00    	m.nop
   2157a: 31 3e d9 12  	l.st.w r25, [r7, 0x138]
   2157e: 01 00        	c.nop
   21580: 00 00 00 00  	v.nop
   21584: 00 00 00 00 00 00    	m.nop
   2158a: 31 3b c0 14  	l.st.w r0, [r7, 0x14c]
   2158e: 01 00        	c.nop
   21590: 00 00 00 00  	v.nop
   21594: 00 00 00 00 00 00    	m.nop
   2159a: 31 3d c0 16  	l.st.w r0, [r7, 0x174]
   2159e: 01 00        	c.nop
   215a0: 00 00 00 00  	v.nop
   215a4: 00 00 00 00 00 00    	m.nop
   215aa: 31 3a d9 12  	l.st.w r25, [r7, 0x128]
   215ae: 01 00        	c.nop
   215b0: 00 00 00 00  	v.nop
   215b4: 00 00 00 00 00 00    	m.nop
   215ba: 31 3f d9 12  	l.st.w r25, [r7, 0x13c]
   215be: 01 00        	c.nop
   215c0: 00 00 00 00  	v.nop
   215c4: 00 00 00 00 00 00    	m.nop
   215ca: 31 3c c0 14  	l.st.w r0, [r7, 0x150]
   215ce: 01 00        	c.nop
   215d0: 00 00 00 00  	v.nop
   215d4: 00 00 00 00 00 00    	m.nop
   215da: 31 3e c0 16  	l.st.w r0, [r7, 0x178]
   215de: 01 00        	c.nop
   215e0: 00 00 00 00  	v.nop
   215e4: 00 00 00 00 00 00    	m.nop
   215ea: 31 38 d0 14  	l.st.w r16, [r7, 0x140]
   215ee: 01 00        	c.nop
   215f0: 00 00 00 00  	v.nop
   215f4: 00 00 00 00 00 00    	m.nop
   215fa: 31 3d c4 14  	l.st.w r4, [r7, 0x154]
   215fe: 01 00        	c.nop
   21600: 00 00 00 00  	v.nop
   21604: 00 00 00 00 00 00    	m.nop
   2160a: 31 3f ca 16  	l.st.w r10, [r7, 0x17c]
   2160e: 01 00        	c.nop
   21610: 00 00 00 00  	v.nop
   21614: 00 00 00 00 00 00    	m.nop
   2161a: 31 3c cc 12  	l.st.w r12, [r7, 0x130]
   2161e: 01 00        	c.nop
   21620: 00 00 00 00  	v.nop
   21624: 00 00 00 00 00 00    	m.nop
   2162a: 31 39 ce 14  	l.st.w r14, [r7, 0x144]
   2162e: 01 00        	c.nop
   21630: 00 00 00 00  	v.nop
   21634: 00 00 00 00 00 00    	m.nop
   2163a: 31 3e c0 14  	l.st.w r0, [r7, 0x158]
   2163e: 01 00        	c.nop
   21640: 00 00 00 00  	v.nop
   21644: 00 00 00 00 00 00    	m.nop
   2164a: 31 38 c0 18  	l.st.w r0, [r7, 0x180]
   2164e: 01 00        	c.nop
   21650: 00 00 00 00  	v.nop
   21654: 00 00 00 00 00 00    	m.nop
   2165a: 31 3d d6 12  	l.st.w r22, [r7, 0x134]
   2165e: 01 00        	c.nop
   21660: 00 00 00 00  	v.nop
   21664: 00 00 00 00 00 00    	m.nop
   2166a: 31 3a c9 14  	l.st.w r9, [r7, 0x148]
   2166e: 01 00        	c.nop
   21670: 00 00 00 00  	v.nop
   21674: 00 00 00 00 00 00    	m.nop
   2167a: 31 3f cb 14  	l.st.w r11, [r7, 0x15c]
   2167e: 01 00        	c.nop
   21680: 00 00 00 00  	v.nop
   21684: 00 00 00 00 00 00    	m.nop
   2168a: 2b 50 85 01  	l.jal r0, 0x5a <0x21c20>
   2168e: 01 00        	c.nop
   21690: 00 00 00 00  	v.nop
   21694: 00 00 00 00 00 00    	m.nop
   2169a: 31 13 48 02  	l.ld.w r12, [sp, 0x28]
   2169e: 05 4a        	<unknown>
   216a0: 00 00 00 00  	v.nop
   216a4: 00 00 00 00 00 00    	m.nop
   216aa: f1 12 40 02  	l.ld.w r11, [sp, 0x20]
   216ae: 01 44        	<unknown>
   216b0: 00 00 00 00  	v.nop
   216b4: 00 00 00 00 00 00    	m.nop
   216ba: b1 4b 4c 00  	l.ld.w r14, [r9, 0xc]
   216be: 12 85        	<unknown>
   216c0: 00 00 00 00  	v.nop
   216c4: 00 00 00 00 00 00    	m.nop
   216ca: b1 14 40 06  	l.ld.w r18, [sp, 0x60]
   216ce: 52 88        	<unknown>
   216d0: 00 00 00 00  	v.nop
   216d4: 00 00 00 00 00 00    	m.nop
   216da: 63 21 9f 03  	l.srli r5, r4, 0x1f
   216de: d2 8c        	<unknown>
   216e0: 00 00 00 00  	v.nop
   216e4: 00 00 00 00 00 00    	m.nop
   216ea: 31 3e c4 14  	l.st.w r4, [r7, 0x158]
   216ee: b2 8b        	<unknown>
   216f0: 00 00 00 00  	v.nop
   216f4: 00 00 00 00 00 00    	m.nop
   216fa: 31 39 d4 12  	l.st.w r20, [r7, 0x124]
   216fe: ae 87        	<unknown>
   21700: 00 00 00 00  	v.nop
   21704: 00 00 00 00 00 00    	m.nop
   2170a: 31 3e d4 12  	l.st.w r20, [r7, 0x138]
   2170e: d2 89        	<unknown>
   21710: 00 00 00 00  	v.nop
   21714: 00 00 00 00 00 00    	m.nop
   2171a: 31 3b c0 14  	l.st.w r0, [r7, 0x14c]
   2171e: 22 82        	<unknown>
   21720: 00 00 00 00  	v.nop
   21724: 00 00 00 00 00 00    	m.nop
   2172a: 31 3d c0 16  	l.st.w r0, [r7, 0x174]
   2172e: c1 04        	<unknown>
   21730: 00 00 00 00  	v.nop
   21734: 00 00 00 00 00 00    	m.nop
   2173a: 31 3a d4 12  	l.st.w r20, [r7, 0x128]
   2173e: 01 00        	c.nop
   21740: 00 00 00 00  	v.nop
   21744: 00 00 00 00 00 00    	m.nop
   2174a: 31 3f d4 12  	l.st.w r20, [r7, 0x13c]
   2174e: 01 00        	c.nop
   21750: 00 00 00 00  	v.nop
   21754: 00 00 00 00 00 00    	m.nop
   2175a: 31 3c c0 14  	l.st.w r0, [r7, 0x150]
   2175e: 01 00        	c.nop
   21760: 00 00 00 00  	v.nop
   21764: 00 00 00 00 00 00    	m.nop
   2176a: 31 3e c0 16  	l.st.w r0, [r7, 0x178]
   2176e: 01 00        	c.nop
   21770: 00 00 00 00  	v.nop
   21774: 00 00 00 00 00 00    	m.nop
   2177a: 31 3b d4 12  	l.st.w r20, [r7, 0x12c]
   2177e: 01 00        	c.nop
   21780: 00 00 00 00  	v.nop
   21784: 00 00 00 00 00 00    	m.nop
   2178a: 31 38 d4 14  	l.st.w r20, [r7, 0x140]
   2178e: 01 00        	c.nop
   21790: 00 00 00 00  	v.nop
   21794: 00 00 00 00 00 00    	m.nop
   2179a: 31 3d c0 14  	l.st.w r0, [r7, 0x154]
   2179e: 01 00        	c.nop
   217a0: 00 00 00 00  	v.nop
   217a4: 00 00 00 00 00 00    	m.nop
   217aa: 31 3f c0 16  	l.st.w r0, [r7, 0x17c]
   217ae: 01 00        	c.nop
   217b0: 00 00 00 00  	v.nop
   217b4: 00 00 00 00 00 00    	m.nop
   217ba: 31 3c cb 12  	l.st.w r11, [r7, 0x130]
   217be: 01 00        	c.nop
   217c0: 00 00 00 00  	v.nop
   217c4: 00 00 00 00 00 00    	m.nop
   217ca: 31 39 ce 14  	l.st.w r14, [r7, 0x144]
   217ce: 01 00        	c.nop
   217d0: 00 00 00 00  	v.nop
   217d4: 00 00 00 00 00 00    	m.nop
   217da: 31 38 c5 18  	l.st.w r5, [r7, 0x180]
   217de: 01 00        	c.nop
   217e0: 00 00 00 00  	v.nop
   217e4: 00 00 00 00 00 00    	m.nop
   217ea: 31 3d cc 12  	l.st.w r12, [r7, 0x134]
   217ee: 01 00        	c.nop
   217f0: 00 00 00 00  	v.nop
   217f4: 00 00 00 00 00 00    	m.nop
   217fa: 71 4a 40 00  	l.ld.w r9, [r9]
   217fe: 01 4e        	<unknown>
   21800: 00 00 00 00  	v.nop
   21804: 00 00 00 00 00 00    	m.nop
   2180a: 31 3f c0 14  	l.st.w r0, [r7, 0x15c]
   2180e: 01 00        	c.nop
   21810: 00 00 00 00  	v.nop
   21814: 00 00 00 00 00 00    	m.nop
   2181a: 31 39 c0 18  	l.st.w r0, [r7, 0x184]
   2181e: 01 00        	c.nop
   21820: 00 00 00 00  	v.nop
   21824: 00 00 00 00 00 00    	m.nop
   2182a: 31 3a c9 14  	l.st.w r9, [r7, 0x148]
   2182e: 01 00        	c.nop
   21830: 00 00 00 00  	v.nop
   21834: 00 00 00 00 00 00    	m.nop
   2183a: bb 50 80 c0  	l.blt.s r10, r0, 0x2 <0x21850>
   2183e: 01 00        	c.nop
   21840: 00 00 00 00  	v.nop
   21844: 00 00 00 00 00 00    	m.nop
   2184a: 00 00        	s.nop16
   2184c: 00 00        	s.nop16
   2184e: 2a 89        	<unknown>
   21850: 00 00 00 00  	v.nop
   21854: 00 00 00 00 00 00    	m.nop
   2185a: 00 00        	s.nop16
   2185c: 00 00        	s.nop16
   2185e: 85 4a        	<unknown>
   21860: 00 00 00 00  	v.nop
   21864: 00 00 00 00 00 00    	m.nop
   2186a: 00 00        	s.nop16
   2186c: 00 00        	s.nop16
   2186e: 72 85        	<unknown>
   21870: 00 00 00 00  	v.nop
   21874: 00 00 00 00 00 00    	m.nop
   2187a: 00 00        	s.nop16
   2187c: 00 00        	s.nop16
   2187e: 72 8d        	<unknown>
   21880: 00 00 00 00  	v.nop
   21884: 00 00 00 00 00 00    	m.nop
   2188a: 00 00        	s.nop16
   2188c: 00 00        	s.nop16
   2188e: f2 8d        	<unknown>
   21890: 00 00 00 00  	v.nop
   21894: 00 00 00 00 00 00    	m.nop
   2189a: 2b 60 83 01  	l.jal r0, 0x3c <0x21c50>
   2189e: 01 00        	c.nop
   218a0: 00 00 00 00  	v.nop
   218a4: 00 00 00 00 00 00    	m.nop
   218aa: f1 15 50 02  	l.ld.w r23, [sp, 0x30]
   218ae: 01 49        	<unknown>
   218b0: 00 00 00 00  	v.nop
   218b4: 00 00 00 00 00 00    	m.nop
   218ba: f1 12 54 02  	l.ld.w r11, [sp, 0x34]
   218be: 4a 8d        	<unknown>
   218c0: 00 00 00 00  	v.nop
   218c4: 00 00 00 00 00 00    	m.nop
   218ca: 71 17 40 04  	l.ld.w r29, [sp, 0x40]
   218ce: ca 82        	<unknown>
   218d0: 00 00 00 00  	v.nop
   218d4: 00 00 00 00 00 00    	m.nop
   218da: 31 13 48 02  	l.ld.w r12, [sp, 0x28]
   218de: 01 00        	c.nop
   218e0: 00 00 00 00  	v.nop
   218e4: 00 00 00 00 00 00    	m.nop
   218ea: 31 12 40 02  	l.ld.w r8, [sp, 0x20]
   218ee: 4a 8b        	<unknown>
   218f0: 00 00 00 00  	v.nop
   218f4: 00 00 00 00 00 00    	m.nop
   218fa: 71 4c 4c 00  	l.ld.w r17, [r9, 0xc]
   218fe: 5e 8a        	<unknown>
   21900: 00 00 00 00  	v.nop
   21904: 00 00 00 00 00 00    	m.nop
   2190a: f1 4c 50 00  	l.ld.w r19, [r9, 0x10]
   2190e: ae 87        	<unknown>
   21910: 00 00 00 00  	v.nop
   21914: 00 00 00 00 00 00    	m.nop
   2191a: 31 4c 54 00  	l.ld.w r16, [r9, 0x14]
   2191e: 76 85        	<unknown>
   21920: 00 00 00 00  	v.nop
   21924: 00 00 00 00 00 00    	m.nop
   2192a: b1 4b 58 00  	l.ld.w r14, [r9, 0x18]
   2192e: b2 8c        	<unknown>
   21930: 00 00 00 00  	v.nop
   21934: 00 00 00 00 00 00    	m.nop
   2193a: 71 4a 5c 00  	l.ld.w r9, [r9, 0x1c]
   2193e: a2 8a        	<unknown>
   21940: 00 00 00 00  	v.nop
   21944: 00 00 00 00 00 00    	m.nop
   2194a: 23 26 9f 03  	l.srli r24, r4, 0x1f
   2194e: 01 00        	c.nop
   21950: 00 00 00 00  	v.nop
   21954: 00 00 00 00 00 00    	m.nop
   2195a: 31 3b c4 14  	l.st.w r4, [r7, 0x14c]
   2195e: 01 00        	c.nop
   21960: 00 00 00 00  	v.nop
   21964: 00 00 00 00 00 00    	m.nop
   2196a: 31 11 58 02  	l.ld.w r4, [sp, 0x38]
   2196e: 01 00        	c.nop
   21970: 00 00 00 00  	v.nop
   21974: 00 00 00 00 00 00    	m.nop
   2197a: 31 3b d7 12  	l.st.w r23, [r7, 0x12c]
   2197e: 85 4b        	<unknown>
   21980: 00 00 00 00  	v.nop
   21984: 00 00 00 00 00 00    	m.nop
   2198a: 31 39 c8 12  	l.st.w r8, [r7, 0x124]
   2198e: 4a 84        	<unknown>
   21990: 00 00 00 00  	v.nop
   21994: 00 00 00 00 00 00    	m.nop
   2199a: f1 16 48 08  	l.ld.w r27, [sp, 0x88]
   2199e: 01 00        	c.nop
   219a0: 00 00 00 00  	v.nop
   219a4: 00 00 00 00 00 00    	m.nop
   219aa: 31 3e d1 12  	l.st.w r17, [r7, 0x138]
   219ae: 01 00        	c.nop
   219b0: 00 00 00 00  	v.nop
   219b4: 00 00 00 00 00 00    	m.nop
   219ba: 31 3d d8 16  	l.st.w r24, [r7, 0x174]
   219be: 01 00        	c.nop
   219c0: 00 00 00 00  	v.nop
   219c4: 00 00 00 00 00 00    	m.nop
   219ca: 31 3a cc 12  	l.st.w r12, [r7, 0x128]
   219ce: 01 00        	c.nop
   219d0: 00 00 00 00  	v.nop
   219d4: 00 00 00 00 00 00    	m.nop
   219da: 31 3f d3 12  	l.st.w r19, [r7, 0x13c]
   219de: 01 00        	c.nop
   219e0: 00 00 00 00  	v.nop
   219e4: 00 00 00 00 00 00    	m.nop
   219ea: 31 3c c0 14  	l.st.w r0, [r7, 0x150]
   219ee: 01 00        	c.nop
   219f0: 00 00 00 00  	v.nop
   219f4: 00 00 00 00 00 00    	m.nop
   219fa: 31 3e c0 16  	l.st.w r0, [r7, 0x178]
   219fe: 01 00        	c.nop
   21a00: 00 00 00 00  	v.nop
   21a04: 00 00 00 00 00 00    	m.nop
   21a0a: 31 38 d0 14  	l.st.w r16, [r7, 0x140]
   21a0e: 01 00        	c.nop
   21a10: 00 00 00 00  	v.nop
   21a14: 00 00 00 00 00 00    	m.nop
   21a1a: 31 3d c4 14  	l.st.w r4, [r7, 0x154]
   21a1e: 01 00        	c.nop
   21a20: 00 00 00 00  	v.nop
   21a24: 00 00 00 00 00 00    	m.nop
   21a2a: 31 3f dd 16  	l.st.w r29, [r7, 0x17c]
   21a2e: 01 00        	c.nop
   21a30: 00 00 00 00  	v.nop
   21a34: 00 00 00 00 00 00    	m.nop
   21a3a: 31 3c cb 12  	l.st.w r11, [r7, 0x130]
   21a3e: 01 00        	c.nop
   21a40: 00 00 00 00  	v.nop
   21a44: 00 00 00 00 00 00    	m.nop
   21a4a: 31 39 ce 14  	l.st.w r14, [r7, 0x144]
   21a4e: 01 00        	c.nop
   21a50: 00 00 00 00  	v.nop
   21a54: 00 00 00 00 00 00    	m.nop
   21a5a: 31 3e c0 14  	l.st.w r0, [r7, 0x158]
   21a5e: 01 00        	c.nop
   21a60: 00 00 00 00  	v.nop
   21a64: 00 00 00 00 00 00    	m.nop
   21a6a: 31 38 c0 18  	l.st.w r0, [r7, 0x180]
   21a6e: 01 00        	c.nop
   21a70: 00 00 00 00  	v.nop
   21a74: 00 00 00 00 00 00    	m.nop
   21a7a: 31 3d d7 12  	l.st.w r23, [r7, 0x134]
   21a7e: 01 00        	c.nop
   21a80: 00 00 00 00  	v.nop
   21a84: 00 00 00 00 00 00    	m.nop
   21a8a: 31 3a c9 14  	l.st.w r9, [r7, 0x148]
   21a8e: 01 00        	c.nop
   21a90: 00 00 00 00  	v.nop
   21a94: 00 00 00 00 00 00    	m.nop
   21a9a: 31 39 c0 18  	l.st.w r0, [r7, 0x184]
   21a9e: 01 00        	c.nop
   21aa0: 00 00 00 00  	v.nop
   21aa4: 00 00 00 00 00 00    	m.nop
   21aaa: 2b 70 81 01  	l.jal r0, 0x1e <0x21c80>
   21aae: 01 00        	c.nop
   21ab0: 00 00 00 00  	v.nop
   21ab4: 00 00 00 00 00 00    	m.nop
   21aba: 31 39 5c 14  	l.ld.w r4, [r7, 0x15c]
   21abe: 01 00        	c.nop
   21ac0: 00 00 00 00  	v.nop
   21ac4: 00 00 00 00 00 00    	m.nop
   21aca: 71 39 58 14  	l.ld.w r5, [r7, 0x158]
   21ace: 01 00        	c.nop
   21ad0: 00 00 00 00  	v.nop
   21ad4: 00 00 00 00 00 00    	m.nop
   21ada: f1 3d 54 12  	l.ld.w r23, [r7, 0x134]
   21ade: 01 00        	c.nop
   21ae0: 00 00 00 00  	v.nop
   21ae4: 00 00 00 00 00 00    	m.nop
   21aea: 71 3a 48 14  	l.ld.w r9, [r7, 0x148]
   21aee: 01 00        	c.nop
   21af0: 00 00 00 00  	v.nop
   21af4: 00 00 00 00 00 00    	m.nop
   21afa: b1 3e 44 18  	l.ld.w r26, [r7, 0x184]
   21afe: 01 00        	c.nop
   21b00: 00 00 00 00  	v.nop
   21b04: 00 00 00 00 00 00    	m.nop
   21b0a: a3 24 9f 0b  	l.srai r18, r4, 0x1f
   21b0e: 01 00        	c.nop
   21b10: 00 00 00 00  	v.nop
   21b14: 00 00 00 00 00 00    	m.nop
   21b1a: 63 25 12 18  	<unknown>
   21b1e: 01 00        	c.nop
   21b20: 00 00 00 00  	v.nop
   21b24: 00 00 00 00 00 00    	m.nop
   21b2a: a3 2d 9f 0b  	l.srai r22, r5, 0x1f
   21b2e: 01 00        	c.nop
   21b30: 00 00 00 00  	v.nop
   21b34: 00 00 00 00 00 00    	m.nop
   21b3a: 23 2e 16 18  	<unknown>
   21b3e: 01 00        	c.nop
   21b40: 00 00 00 00  	v.nop
   21b44: 00 00 00 00 00 00    	m.nop
   21b4a: 25 af 92 02  	l.xorix r28, r21, 0x12
   21b4e: 01 00        	c.nop
   21b50: 00 00 00 00  	v.nop
   21b54: 00 00 00 00 00 00    	m.nop
   21b5a: f1 3b 50 12  	l.ld.w r15, [r7, 0x130]
   21b5e: 01 00        	c.nop
   21b60: 00 00 00 00  	v.nop
   21b64: 00 00 00 00 00 00    	m.nop
   21b6a: b1 3b 44 14  	l.ld.w r14, [r7, 0x144]
   21b6e: 01 00        	c.nop
   21b70: 00 00 00 00  	v.nop
   21b74: 00 00 00 00 00 00    	m.nop
   21b7a: 71 39 40 18  	l.ld.w r5, [r7, 0x180]
   21b7e: 01 00        	c.nop
   21b80: 00 00 00 00  	v.nop
   21b84: 00 00 00 00 00 00    	m.nop
   21b8a: 31 3d 4c 12  	l.ld.w r20, [r7, 0x12c]
   21b8e: 01 00        	c.nop
   21b90: 00 00 00 00  	v.nop
   21b94: 00 00 00 00 00 00    	m.nop
   21b9a: 31 3c 40 14  	l.ld.w r16, [r7, 0x140]
   21b9e: 01 00        	c.nop
   21ba0: 00 00 00 00  	v.nop
   21ba4: 00 00 00 00 00 00    	m.nop
   21baa: 31 39 54 14  	l.ld.w r4, [r7, 0x154]
   21bae: 01 00        	c.nop
   21bb0: 00 00 00 00  	v.nop
   21bb4: 00 00 00 00 00 00    	m.nop
   21bba: b1 3a 5c 16  	l.ld.w r10, [r7, 0x17c]
   21bbe: 01 00        	c.nop
   21bc0: 00 00 00 00  	v.nop
   21bc4: 00 00 00 00 00 00    	m.nop
   21bca: 71 3e 48 12  	l.ld.w r25, [r7, 0x128]
   21bce: 01 00        	c.nop
   21bd0: 00 00 00 00  	v.nop
   21bd4: 00 00 00 00 00 00    	m.nop
   21bda: f1 3c 5c 12  	l.ld.w r19, [r7, 0x13c]
   21bde: 01 00        	c.nop
   21be0: 00 00 00 00  	v.nop
   21be4: 00 00 00 00 00 00    	m.nop
   21bea: 71 3c 50 14  	l.ld.w r17, [r7, 0x150]
   21bee: 01 00        	c.nop
   21bf0: 00 00 00 00  	v.nop
   21bf4: 00 00 00 00 00 00    	m.nop
   21bfa: 31 3a 58 16  	l.ld.w r8, [r7, 0x178]
   21bfe: 01 00        	c.nop
   21c00: 00 00 00 00  	v.nop
   21c04: 00 00 00 00 00 00    	m.nop
   21c0a: 71 3d 44 12  	l.ld.w r21, [r7, 0x124]
   21c0e: 01 00        	c.nop
   21c10: 00 00 00 00  	v.nop
   21c14: 00 00 00 00 00 00    	m.nop
   21c1a: a5 c4 96 02  	l.xorix r18, r24, 0x16
   21c1e: 01 00        	c.nop
   21c20: 00 00 00 00  	v.nop
   21c24: 00 00 00 00 00 00    	m.nop
   21c2a: a3 25 9f 0b  	l.srai r22, r4, 0x1f
   21c2e: 01 00        	c.nop
   21c30: 00 00 00 00  	v.nop
   21c34: 00 00 00 00 00 00    	m.nop
   21c3a: 23 26 16 18  	<unknown>
   21c3e: 46 82        	<unknown>
   21c40: 00 00 00 00  	v.nop
   21c44: 00 00 00 00 00 00    	m.nop
   21c4a: e5 c6 96 02  	l.xorix r27, r24, 0x16
   21c4e: 01 00        	c.nop
   21c50: 00 00 00 00  	v.nop
   21c54: 00 00 00 00 00 00    	m.nop
   21c5a: 63 24 9f 0b  	l.srai r17, r4, 0x1f
   21c5e: 01 00        	c.nop
   21c60: 00 00 00 00  	v.nop
   21c64: 00 00 00 00 00 00    	m.nop
   21c6a: 00 00        	s.nop16
   21c6c: 00 00        	s.nop16
   21c6e: 46 92        	<unknown>
   21c70: 00 00 00 00  	v.nop
   21c74: 00 00 00 00 00 00    	m.nop
   21c7a: a5 25 91 02  	l.xorix r22, r4, 0x11
   21c7e: 01 00        	c.nop
   21c80: 00 00 00 00  	v.nop
   21c84: 00 00 00 00 00 00    	m.nop
   21c8a: ef ef 7f 1f  	l.ldi16.u r29, 0xffff
   21c8e: 01 00        	c.nop
   21c90: 00 00 00 00  	v.nop
   21c94: 00 00 00 00 00 00    	m.nop
   21c9a: ef ef 87 00  	l.ldi_hi r29, 0xff
   21c9e: 01 00        	c.nop
   21ca0: 00 00 00 00  	v.nop
   21ca4: 00 00 00 00 00 00    	m.nop
   21caa: 65 bc 9d 01  	l.andix r17, r23, 0x1d
   21cae: 01 00        	c.nop
   21cb0: 00 00 00 00  	v.nop
   21cb4: 00 00 00 00 00 00    	m.nop
   21cba: f1 12 4c 02  	l.ld.w r11, [sp, 0x2c]
   21cbe: e2 02        	<unknown>
   21cc0: 00 00 00 00  	v.nop
   21cc4: 00 00 00 00 00 00    	m.nop
   21cca: 23 d1 18 03  	l.slli r4, r26, 0x18
   21cce: 62 05        	<unknown>
   21cd0: 00 00 00 00  	v.nop
   21cd4: 00 00 00 00 00 00    	m.nop
   21cda: e5 7b 9d 01  	l.andix r15, r15, 0x1d
   21cde: 62 04        	<unknown>
   21ce0: 00 00 00 00  	v.nop
   21ce4: 00 00 00 00 00 00    	m.nop
   21cea: 25 a5 9d 01  	l.andix r20, r20, 0x1d
   21cee: 71 46        	<unknown>
   21cf0: 00 00 00 00  	v.nop
   21cf4: 00 00 00 00 00 00    	m.nop
   21cfa: 65 ce 9d 01  	l.andix r25, r25, 0x1d
   21cfe: 01 00        	c.nop
   21d00: 00 00 00 00  	v.nop
   21d04: 00 00 00 00 00 00    	m.nop
   21d0a: a5 2e 8b 01  	l.andix r26, r5, 0xb
   21d0e: 6d 8d        	<unknown>
   21d10: 00 00 00 00  	v.nop
   21d14: 00 00 00 00 00 00    	m.nop
   21d1a: 71 39 58 08  	l.ld.w r5, [r7, 0x98]
   21d1e: 6d 8c        	<unknown>
   21d20: 00 00 00 00  	v.nop
   21d24: 00 00 00 00 00 00    	m.nop
   21d2a: 25 26 8b 01  	l.andix r24, r4, 0xb
   21d2e: 01 00        	c.nop
   21d30: 00 00 00 00  	v.nop
   21d34: 00 00 00 00 00 00    	m.nop
   21d3a: e5 e2 9d 01  	l.andix r11, r28, 0x1d
   21d3e: 01 00        	c.nop
   21d40: 00 00 00 00  	v.nop
   21d44: 00 00 00 00 00 00    	m.nop
   21d4a: 23 4f 1c 18  	<unknown>
   21d4e: 01 00        	c.nop
   21d50: 00 00 00 00  	v.nop
   21d54: 00 00 00 00 00 00    	m.nop
   21d5a: e5 c2 0b 02  	l.orix r11, r24, 0xb
   21d5e: 01 00        	c.nop
   21d60: 00 00 00 00  	v.nop
   21d64: 00 00 00 00 00 00    	m.nop
   21d6a: 23 86 1b 18  	<unknown>
   21d6e: 01 00        	c.nop
   21d70: 00 00 00 00  	v.nop
   21d74: 00 00 00 00 00 00    	m.nop
   21d7a: 25 c6 1c 02  	l.orix r24, r24, 0x1c
   21d7e: 01 00        	c.nop
   21d80: 00 00 00 00  	v.nop
   21d84: 00 00 00 00 00 00    	m.nop
   21d8a: 23 97 0e 18  	<unknown>
   21d8e: 01 00        	c.nop
   21d90: 00 00 00 00  	v.nop
   21d94: 00 00 00 00 00 00    	m.nop
   21d9a: 25 c6 1c 02  	l.orix r24, r24, 0x1c
   21d9e: 01 00        	c.nop
   21da0: 00 00 00 00  	v.nop
   21da4: 00 00 00 00 00 00    	m.nop
   21daa: 25 b7 9d 01  	l.andix r28, r22, 0x1d
   21dae: 4e 9b        	<unknown>
   21db0: 00 00 00 00  	v.nop
   21db4: 00 00 00 00 00 00    	m.nop
   21dba: a5 c5 16 02  	l.orix r22, r24, 0x16
   21dbe: 01 00        	c.nop
   21dc0: 00 00 00 00  	v.nop
   21dc4: 00 00 00 00 00 00    	m.nop
   21dca: 23 2e 0c 10  	<unknown>
   21dce: 01 00        	c.nop
   21dd0: 00 00 00 00  	v.nop
   21dd4: 00 00 00 00 00 00    	m.nop
   21dda: 31 13 58 04  	l.ld.w r12, [sp, 0x58]
   21dde: 01 00        	c.nop
   21de0: 00 00 00 00  	v.nop
   21de4: 00 00 00 00 00 00    	m.nop
   21dea: ef 27 7f 1f  	l.ldi16.u r4, 0xffff
   21dee: 01 00        	c.nop
   21df0: 00 00 00 00  	v.nop
   21df4: 00 00 00 00 00 00    	m.nop
   21dfa: e5 de 9d 01  	l.andix r27, r27, 0x1d
   21dfe: 01 00        	c.nop
   21e00: 00 00 00 00  	v.nop
   21e04: 00 00 00 00 00 00    	m.nop
   21e0a: a5 94 9d 01  	l.andix r18, r18, 0x1d
   21e0e: 01 00        	c.nop
   21e10: 00 00 00 00  	v.nop
   21e14: 00 00 00 00 00 00    	m.nop
   21e1a: 65 6b 84 01  	l.andix r13, r13, 0x4
   21e1e: 01 00        	c.nop
   21e20: 00 00 00 00  	v.nop
   21e24: 00 00 00 00 00 00    	m.nop
   21e2a: 31 c0 cc 00  	l.st.w r12, [r24]
   21e2e: 01 00        	c.nop
   21e30: 00 00 00 00  	v.nop
   21e34: 00 00 00 00 00 00    	m.nop
   21e3a: 31 13 50 12  	l.ld.w r12, [sp, 0x130]
   21e3e: 01 00        	c.nop
   21e40: 00 00 00 00  	v.nop
   21e44: 00 00 00 00 00 00    	m.nop
   21e4a: 65 ad 9d 01  	l.andix r21, r21, 0x1d
   21e4e: 01 00        	c.nop
   21e50: 00 00 00 00  	v.nop
   21e54: 00 00 00 00 00 00    	m.nop
   21e5a: a5 52 1b 02  	l.orix r10, r10, 0x1b
   21e5e: 01 00        	c.nop
   21e60: 00 00 00 00  	v.nop
   21e64: 00 00 00 00 00 00    	m.nop
   21e6a: 25 e2 08 02  	l.orix r8, r28, 0x8
   21e6e: 01 00        	c.nop
   21e70: 00 00 00 00  	v.nop
   21e74: 00 00 00 00 00 00    	m.nop
   21e7a: 65 4a 9d 01  	l.andix r9, r9, 0x1d
   21e7e: 01 00        	c.nop
   21e80: 00 00 00 00  	v.nop
   21e84: 00 00 00 00 00 00    	m.nop
   21e8a: 31 c7 cc fe  	l.st.w r12, [r24, -0x4]
   21e8e: 01 00        	c.nop
   21e90: 00 00 00 00  	v.nop
   21e94: 00 00 00 00 00 00    	m.nop
   21e9a: 31 13 40 0c  	l.ld.w r12, [sp, 0xc0]
   21e9e: 01 00        	c.nop
   21ea0: 00 00 00 00  	v.nop
   21ea4: 00 00 00 00 00 00    	m.nop
   21eaa: 2f b8 80 30  	l.ldi_hi.m1 r23, 0x8000
   21eae: 01 00        	c.nop
   21eb0: 00 00 00 00  	v.nop
   21eb4: 00 00 00 00 00 00    	m.nop
   21eba: 21 31 98 00  	l.addi.u r4, r6, 0x98
   21ebe: 01 00        	c.nop
   21ec0: 00 00 00 00  	v.nop
   21ec4: 00 00 00 00 00 00    	m.nop
   21eca: 31 c2 cc 00  	l.st.w r12, [r24, 0x8]
   21ece: 01 00        	c.nop
   21ed0: 00 00 00 00  	v.nop
   21ed4: 00 00 00 00 00 00    	m.nop
   21eda: 25 93 1a 02  	l.orix r12, r18, 0x1a
   21ede: 01 00        	c.nop
   21ee0: 00 00 00 00  	v.nop
   21ee4: 00 00 00 00 00 00    	m.nop
   21eea: 31 c1 cd 00  	l.st.w r13, [r24, 0x4]
   21eee: 01 00        	c.nop
   21ef0: 00 00 00 00  	v.nop
   21ef4: 00 00 00 00 00 00    	m.nop
   21efa: 31 c2 d1 0a  	l.st.w r17, [r24, 0xa8]
   21efe: 01 00        	c.nop
   21f00: 00 00 00 00  	v.nop
   21f04: 00 00 00 00 00 00    	m.nop
   21f0a: 31 c3 cf 0a  	l.st.w r15, [r24, 0xac]
   21f0e: 01 00        	c.nop
   21f10: 00 00 00 00  	v.nop
   21f14: 00 00 00 00 00 00    	m.nop
   21f1a: 31 c4 d4 0a  	l.st.w r20, [r24, 0xb0]
   21f1e: 01 00        	c.nop
   21f20: 00 00 00 00  	v.nop
   21f24: 00 00 00 00 00 00    	m.nop
   21f2a: 31 c5 d9 0a  	l.st.w r25, [r24, 0xb4]
   21f2e: 01 00        	c.nop
   21f30: 00 00 00 00  	v.nop
   21f34: 00 00 00 00 00 00    	m.nop
   21f3a: 31 c4 d5 00  	l.st.w r21, [r24, 0x10]
   21f3e: 01 00        	c.nop
   21f40: 00 00 00 00  	v.nop
   21f44: 00 00 00 00 00 00    	m.nop
   21f4a: 31 c1 cb 0c  	l.st.w r11, [r24, 0xc4]
   21f4e: 01 00        	c.nop
   21f50: 00 00 00 00  	v.nop
   21f54: 00 00 00 00 00 00    	m.nop
   21f5a: 31 c2 cc 0c  	l.st.w r12, [r24, 0xc8]
   21f5e: 01 00        	c.nop
   21f60: 00 00 00 00  	v.nop
   21f64: 00 00 00 00 00 00    	m.nop
   21f6a: 31 c3 ca 0c  	l.st.w r10, [r24, 0xcc]
   21f6e: 01 00        	c.nop
   21f70: 00 00 00 00  	v.nop
   21f74: 00 00 00 00 00 00    	m.nop
   21f7a: 31 c1 c8 06  	l.st.w r8, [r24, 0x64]
   21f7e: 01 00        	c.nop
   21f80: 00 00 00 00  	v.nop
   21f84: 00 00 00 00 00 00    	m.nop
   21f8a: 25 72 9d 01  	l.andix r8, r14, 0x1d
   21f8e: 01 00        	c.nop
   21f90: 00 00 00 00  	v.nop
   21f94: 00 00 00 00 00 00    	m.nop
   21f9a: 31 c1 c9 0a  	l.st.w r9, [r24, 0xa4]
   21f9e: 01 00        	c.nop
   21fa0: 00 00 00 00  	v.nop
   21fa4: 00 00 00 00 00 00    	m.nop
   21faa: 31 c6 c8 0a  	l.st.w r8, [r24, 0xb8]
   21fae: 01 00        	c.nop
   21fb0: 00 00 00 00  	v.nop
   21fb4: 00 00 00 00 00 00    	m.nop
   21fba: 65 82 9d 01  	l.andix r9, r16, 0x1d
   21fbe: 01 00        	c.nop
   21fc0: 00 00 00 00  	v.nop
   21fc4: 00 00 00 00 00 00    	m.nop
   21fca: 25 9a 9d 01  	l.andix r8, r19, 0x1d
   21fce: 01 00        	c.nop
   21fd0: 00 00 00 00  	v.nop
   21fd4: 00 00 00 00 00 00    	m.nop
   21fda: 31 c7 c9 0a  	l.st.w r9, [r24, 0xbc]
   21fde: 01 00        	c.nop
   21fe0: 00 00 00 00  	v.nop
   21fe4: 00 00 00 00 00 00    	m.nop
   21fea: 31 c0 c8 0c  	l.st.w r8, [r24, 0xc0]
   21fee: 01 44        	<unknown>
   21ff0: 00 00 00 00  	v.nop
   21ff4: 00 00 00 00 00 00    	m.nop
   21ffa: 2f 48 88 20  	l.ldi_hi.m1 r9, 0x100
   21ffe: 01 00        	c.nop
   22000: 00 00 00 00  	v.nop
   22004: 00 00 00 00 00 00    	m.nop
   2200a: 31 c4 c0 0c  	l.st.w r0, [r24, 0xd0]
   2200e: 01 00        	c.nop
   22010: 00 00 00 00  	v.nop
   22014: 00 00 00 00 00 00    	m.nop
   2201a: bb b0 09 c0  	l.blt.u r22, r9, 0x2 <0x22030>
   2201e: 01 00        	c.nop
   22020: 00 00 00 00  	v.nop
   22024: 00 00 00 00 00 00    	m.nop
   2202a: 00 00        	s.nop16
   2202c: 00 00        	s.nop16
   2202e: a2 8b        	<unknown>
   22030: 00 00 00 00  	v.nop
   22034: 00 00 00 00 00 00    	m.nop
   2203a: 31 12 48 0c  	l.ld.w r8, [sp, 0xc8]
   2203e: 05 4d        	<unknown>
   22040: 00 00 00 00  	v.nop
   22044: 00 00 00 00 00 00    	m.nop
   2204a: 2f 48 18 21  	l.ldi16.s r9, 0x300
   2204e: 81 47        	<unknown>
   22050: 00 00 00 00  	v.nop
   22054: 00 00 00 00 00 00    	m.nop
   2205a: ef e7 07 21  	l.ldi16.s r28, 0xff
   2205e: 01 00        	c.nop
   22060: 00 00 00 00  	v.nop
   22064: 00 00 00 00 00 00    	m.nop
   2206a: af ef 80 20  	l.ldi_hi.m1 r29, 0x1e
   2206e: 01 00        	c.nop
   22070: 00 00 00 00  	v.nop
   22074: 00 00 00 00 00 00    	m.nop
   2207a: 00 00        	s.nop16
   2207c: 00 00        	s.nop16
   2207e: 1e 04        	<unknown>
   22080: 00 00 00 00  	v.nop
   22084: 00 00 00 00 00 00    	m.nop
   2208a: 71 12 44 0c  	l.ld.w r9, [sp, 0xc4]
   2208e: 65 8c        	<unknown>
   22090: 00 00 00 00  	v.nop
   22094: 00 00 00 00 00 00    	m.nop
   2209a: 25 42 17 02  	l.orix r8, r8, 0x17
   2209e: 01 00        	c.nop
   220a0: 00 00 00 00  	v.nop
   220a4: 00 00 00 00 00 00    	m.nop
   220aa: 00 00        	s.nop16
   220ac: 00 00        	s.nop16
   220ae: 45 8c        	<unknown>
   220b0: 00 00 00 00  	v.nop
   220b4: 00 00 00 00 00 00    	m.nop
   220ba: 00 00        	s.nop16
   220bc: 00 00        	s.nop16
   220be: d1 44        	<unknown>
   220c0: 00 00 00 00  	v.nop
   220c4: 00 00 00 00 00 00    	m.nop
   220ca: 00 00        	s.nop16
   220cc: 00 00        	s.nop16
   220ce: a6 92        	<unknown>
   220d0: 00 00 00 00  	v.nop
   220d4: 00 00 00 00 00 00    	m.nop
   220da: 31 28 c8 00  	l.st.w r8, [r5]
   220de: 01 00        	c.nop
   220e0: 00 00 00 00  	v.nop
   220e4: 00 00 00 00 00 00    	m.nop
   220ea: 31 22 40 00  	l.ld.w r8, [r4]
   220ee: 01 00        	c.nop
   220f0: 00 00 00 00  	v.nop
   220f4: 00 00 00 00 00 00    	m.nop
   220fa: 71 31 50 00  	l.ld.w r5, [r6, 0x10]
   220fe: 01 00        	c.nop
   22100: 00 00 00 00  	v.nop
   22104: 00 00 00 00 00 00    	m.nop
   2210a: 71 15 50 08  	l.ld.w r21, [sp, 0x90]
   2210e: 01 00        	c.nop
   22110: 00 00 00 00  	v.nop
   22114: 00 00 00 00 00 00    	m.nop
   2211a: 31 16 44 12  	l.ld.w r24, [sp, 0x124]
   2211e: 01 00        	c.nop
   22120: 00 00 00 00  	v.nop
   22124: 00 00 00 00 00 00    	m.nop
   2212a: b1 14 54 08  	l.ld.w r18, [sp, 0x94]
   2212e: 01 00        	c.nop
   22130: 00 00 00 00  	v.nop
   22134: 00 00 00 00 00 00    	m.nop
   2213a: f1 16 40 12  	l.ld.w r27, [sp, 0x120]
   2213e: 01 00        	c.nop
   22140: 00 00 00 00  	v.nop
   22144: 00 00 00 00 00 00    	m.nop
   2214a: 71 16 50 06  	l.ld.w r25, [sp, 0x70]
   2214e: 01 00        	c.nop
   22150: 00 00 00 00  	v.nop
   22154: 00 00 00 00 00 00    	m.nop
   2215a: 21 31 10 00  	l.addi.u r4, r6, 0x10
   2215e: 01 00        	c.nop
   22160: 00 00 00 00  	v.nop
   22164: 00 00 00 00 00 00    	m.nop
   2216a: 3b 2a 3a 00  	<unknown>
   2216e: 01 00        	c.nop
   22170: 00 00 00 00  	v.nop
   22174: 00 00 00 00 00 00    	m.nop
   2217a: 31 15 4c 14  	l.ld.w r20, [sp, 0x14c]
   2217e: 09 4b        	<unknown>
   22180: 00 00 00 00  	v.nop
   22184: 00 00 00 00 00 00    	m.nop
   2218a: 71 14 48 14  	l.ld.w r17, [sp, 0x148]
   2218e: 01 00        	c.nop
   22190: 00 00 00 00  	v.nop
   22194: 00 00 00 00 00 00    	m.nop
   2219a: f1 14 44 14  	l.ld.w r19, [sp, 0x144]
   2219e: 01 00        	c.nop
   221a0: 00 00 00 00  	v.nop
   221a4: 00 00 00 00 00 00    	m.nop
   221aa: fb 2c 00 40  	<unknown>
   221ae: 01 00        	c.nop
   221b0: 00 00 00 00  	v.nop
   221b4: 00 00 00 00 00 00    	m.nop
   221ba: 73 b9 83 00  	l.movs r5, md23_wrk
   221be: 01 00        	c.nop
   221c0: 00 00 00 00  	v.nop
   221c4: 00 00 00 00 00 00    	m.nop
   221ca: 65 29 96 01  	l.andix r5, r5, 0x16
   221ce: 01 00        	c.nop
   221d0: 00 00 00 00  	v.nop
   221d4: 00 00 00 00 00 00    	m.nop
   221da: fb 2b 00 00  	<unknown>
   221de: 01 00        	c.nop
   221e0: 00 00 00 00  	v.nop
   221e4: 00 00 00 00 00 00    	m.nop
   221ea: 2f 2a 04 21  	l.ldi16.s r5, 0x88
   221ee: 01 00        	c.nop
   221f0: 00 00 00 00  	v.nop
   221f4: 00 00 00 00 00 00    	m.nop
   221fa: 63 41 05 10  	<unknown>
   221fe: 01 00        	c.nop
   22200: 00 00 00 00  	v.nop
   22204: 00 00 00 00 00 00    	m.nop
   2220a: 71 2a 40 00  	l.ld.w r9, [r5]
   2220e: 01 00        	c.nop
   22210: 00 00 00 00  	v.nop
   22214: 00 00 00 00 00 00    	m.nop
   2221a: b3 a2 83 00  	l.movs r10, md20_wrk
   2221e: 01 00        	c.nop
   22220: 00 00 00 00  	v.nop
   22224: 00 00 00 00 00 00    	m.nop
   2222a: 2f 60 f0 3e  	l.ldi_hi.m1 r12, 0xfe00
   2222e: 06 05        	<unknown>
   22230: 00 00 00 00  	v.nop
   22234: 00 00 00 00 00 00    	m.nop
   2223a: ef 67 7f 1f  	l.ldi16.u r12, 0xffff
   2223e: 71 8d        	<unknown>
   22240: 00 00 00 00  	v.nop
   22244: 00 00 00 00 00 00    	m.nop
   2224a: f3 02 80 00  	l.movs r11, sip_id
   2224e: 01 00        	c.nop
   22250: 00 00 00 00  	v.nop
   22254: 00 00 00 00 00 00    	m.nop
   2225a: 6f 60 80 00  	l.ldi_hi r12, 0x1
   2225e: be 05        	<unknown>
   22260: 00 00 00 00  	v.nop
   22264: 00 00 00 00 00 00    	m.nop
   2226a: 31 13 4c 02  	l.ld.w r12, [sp, 0x2c]
   2226e: f1 8c        	<unknown>
   22270: 00 00 00 00  	v.nop
   22274: 00 00 00 00 00 00    	m.nop
   2227a: 00 00        	s.nop16
   2227c: 00 00        	s.nop16
   2227e: c9 8c        	<unknown>
   22280: 00 00 00 00  	v.nop
   22284: 00 00 00 00 00 00    	m.nop
   2228a: 00 00        	s.nop16
   2228c: 00 00        	s.nop16
   2228e: f1 8d        	<unknown>
   22290: 00 00 00 00  	v.nop
   22294: 00 00 00 00 00 00    	m.nop
   2229a: 00 00        	s.nop16
   2229c: 00 00        	s.nop16
   2229e: cd 8c        	<unknown>
   222a0: 00 00 00 00  	v.nop
   222a4: 00 00 00 00 00 00    	m.nop
   222aa: 65 4a 1d 02  	l.orix r9, r9, 0x1d
   222ae: 01 00        	c.nop
   222b0: 00 00 00 00  	v.nop
   222b4: 00 00 00 00 00 00    	m.nop
   222ba: 31 28 c9 00  	l.st.w r9, [r5]
   222be: 01 00        	c.nop
   222c0: 00 00 00 00  	v.nop
   222c4: 00 00 00 00 00 00    	m.nop
   222ca: 71 21 40 00  	l.ld.w r5, [r4]
   222ce: 01 00        	c.nop
   222d0: 00 00 00 00  	v.nop
   222d4: 00 00 00 00 00 00    	m.nop
   222da: 3b 2e 1a 00  	<unknown>
   222de: 01 00        	c.nop
   222e0: 00 00 00 00  	v.nop
   222e4: 00 00 00 00 00 00    	m.nop
   222ea: 7b 2b 20 40  	<unknown>
   222ee: 01 00        	c.nop
   222f0: 00 00 00 00  	v.nop
   222f4: 00 00 00 00 00 00    	m.nop
   222fa: 2f 28 08 29  	l.ldi16.s r5, 0x4100
   222fe: 01 00        	c.nop
   22300: 00 00 00 00  	v.nop
   22304: 00 00 00 00 00 00    	m.nop
   2230a: af 2f 91 00  	l.ldi_hi r5, 0x23e
   2230e: 01 00        	c.nop
   22310: 00 00 00 00  	v.nop
   22314: 00 00 00 00 00 00    	m.nop
   2231a: 71 32 00 06  	l.ld.b r9, [r6, 0x60]
   2231e: 01 00        	c.nop
   22320: 00 00 00 00  	v.nop
   22324: 00 00 00 00 00 00    	m.nop
   2232a: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   2232e: 01 00        	c.nop
   22330: 00 00 00 00  	v.nop
   22334: 00 00 00 00 00 00    	m.nop
   2233a: fb 4e 00 00  	<unknown>
   2233e: 01 00        	c.nop
   22340: 00 00 00 00  	v.nop
   22344: 00 00 00 00 00 00    	m.nop
   2234a: b1 32 5c 04  	l.ld.w r10, [r6, 0x5c]
   2234e: 01 00        	c.nop
   22350: 00 00 00 00  	v.nop
   22354: 00 00 00 00 00 00    	m.nop
   2235a: f1 32 58 04  	l.ld.w r11, [r6, 0x58]
   2235e: 01 00        	c.nop
   22360: 00 00 00 00  	v.nop
   22364: 00 00 00 00 00 00    	m.nop
   2236a: 61 32 5c 00  	l.addi.u r9, r6, 0x5c
   2236e: 01 00        	c.nop
   22370: 00 00 00 00  	v.nop
   22374: 00 00 00 00 00 00    	m.nop
   2237a: 7b 52 2b 40  	<unknown>
   2237e: 01 00        	c.nop
   22380: 00 00 00 00  	v.nop
   22384: 00 00 00 00 00 00    	m.nop
   2238a: b1 32 5c 02  	l.ld.w r10, [r6, 0x3c]
   2238e: 01 00        	c.nop
   22390: 00 00 00 00  	v.nop
   22394: 00 00 00 00 00 00    	m.nop
   2239a: f1 32 58 02  	l.ld.w r11, [r6, 0x38]
   2239e: 01 00        	c.nop
   223a0: 00 00 00 00  	v.nop
   223a4: 00 00 00 00 00 00    	m.nop
   223aa: 31 48 cf 00  	l.st.w r15, [r9]
   223ae: 01 00        	c.nop
   223b0: 00 00 00 00  	v.nop
   223b4: 00 00 00 00 00 00    	m.nop
   223ba: 00 00        	s.nop16
   223bc: 00 00        	s.nop16
   223be: 2e 95        	<unknown>
   223c0: 00 00 00 00  	v.nop
   223c4: 00 00 00 00 00 00    	m.nop
   223ca: 31 37 ca 02  	l.st.w r10, [r6, 0x3c]
   223ce: 01 00        	c.nop
   223d0: 00 00 00 00  	v.nop
   223d4: 00 00 00 00 00 00    	m.nop
   223da: 2b 70 82 01  	l.jal r0, 0x2e <0x226b0>
   223de: 01 00        	c.nop
   223e0: 00 00 00 00  	v.nop
   223e4: 00 00 00 00 00 00    	m.nop
   223ea: 73 b9 83 00  	l.movs r5, md23_wrk
   223ee: 91 44        	<unknown>
   223f0: 00 00 00 00  	v.nop
   223f4: 00 00 00 00 00 00    	m.nop
   223fa: 31 15 4c 14  	l.ld.w r20, [sp, 0x14c]
   223fe: 09 4b        	<unknown>
   22400: 00 00 00 00  	v.nop
   22404: 00 00 00 00 00 00    	m.nop
   2240a: 71 14 48 14  	l.ld.w r17, [sp, 0x148]
   2240e: 01 00        	c.nop
   22410: 00 00 00 00  	v.nop
   22414: 00 00 00 00 00 00    	m.nop
   2241a: f1 14 44 14  	l.ld.w r19, [sp, 0x144]
   2241e: 01 00        	c.nop
   22420: 00 00 00 00  	v.nop
   22424: 00 00 00 00 00 00    	m.nop
   2242a: 65 29 89 01  	l.andix r5, r5, 0x9
   2242e: 01 00        	c.nop
   22430: 00 00 00 00  	v.nop
   22434: 00 00 00 00 00 00    	m.nop
   2243a: fb 2e 40 7f  	<unknown>
   2243e: 01 00        	c.nop
   22440: 00 00 00 00  	v.nop
   22444: 00 00 00 00 00 00    	m.nop
   2244a: 2b 40 fe 3f  	l.jal r0, -0x18 <0x222c0>
   2244e: 01 00        	c.nop
   22450: 00 00 00 00  	v.nop
   22454: 00 00 00 00 00 00    	m.nop
   2245a: 73 01 80 00  	l.movs r5, sip_id
   2245e: 01 00        	c.nop
   22460: 00 00 00 00  	v.nop
   22464: 00 00 00 00 00 00    	m.nop
   2246a: ef 4f 80 20  	l.ldi_hi.m1 r9, 0x1f
   2246e: c2 02        	<unknown>
   22470: 00 00 00 00  	v.nop
   22474: 00 00 00 00 00 00    	m.nop
   2247a: 65 29 89 01  	l.andix r5, r5, 0x9
   2247e: 01 00        	c.nop
   22480: 00 00 00 00  	v.nop
   22484: 00 00 00 00 00 00    	m.nop
   2248a: 2f 48 08 15  	l.ldi16.u r9, 0xa100
   2248e: 01 00        	c.nop
   22490: 00 00 00 00  	v.nop
   22494: 00 00 00 00 00 00    	m.nop
   2249a: 2f 48 90 00  	l.ldi_hi r9, 0x200
   2249e: 01 00        	c.nop
   224a0: 00 00 00 00  	v.nop
   224a4: 00 00 00 00 00 00    	m.nop
   224aa: 65 29 09 02  	l.orix r5, r5, 0x9
   224ae: 01 00        	c.nop
   224b0: 00 00 00 00  	v.nop
   224b4: 00 00 00 00 00 00    	m.nop
   224ba: 71 32 00 06  	l.ld.b r9, [r6, 0x60]
   224be: 01 00        	c.nop
   224c0: 00 00 00 00  	v.nop
   224c4: 00 00 00 00 00 00    	m.nop
   224ca: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   224ce: 01 00        	c.nop
   224d0: 00 00 00 00  	v.nop
   224d4: 00 00 00 00 00 00    	m.nop
   224da: fb 49 60 7f  	<unknown>
   224de: 01 00        	c.nop
   224e0: 00 00 00 00  	v.nop
   224e4: 00 00 00 00 00 00    	m.nop
   224ea: 71 32 00 04  	l.ld.b r9, [r6, 0x40]
   224ee: 01 00        	c.nop
   224f0: 00 00 00 00  	v.nop
   224f4: 00 00 00 00 00 00    	m.nop
   224fa: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   224fe: 01 00        	c.nop
   22500: 00 00 00 00  	v.nop
   22504: 00 00 00 00 00 00    	m.nop
   2250a: bb 4c 20 00  	<unknown>
   2250e: 01 00        	c.nop
   22510: 00 00 00 00  	v.nop
   22514: 00 00 00 00 00 00    	m.nop
   2251a: 71 32 09 02  	l.ld.b r9, [r6, 0x29]
   2251e: 01 00        	c.nop
   22520: 00 00 00 00  	v.nop
   22524: 00 00 00 00 00 00    	m.nop
   2252a: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   2252e: 01 00        	c.nop
   22530: 00 00 00 00  	v.nop
   22534: 00 00 00 00 00 00    	m.nop
   2253a: fb 4e 00 07  	<unknown>
   2253e: 01 00        	c.nop
   22540: 00 00 00 00  	v.nop
   22544: 00 00 00 00 00 00    	m.nop
   2254a: f1 12 40 08  	l.ld.w r11, [sp, 0x80]
   2254e: 01 00        	c.nop
   22550: 00 00 00 00  	v.nop
   22554: 00 00 00 00 00 00    	m.nop
   2255a: 71 5a 40 00  	l.ld.w r9, [r11]
   2255e: 01 00        	c.nop
   22560: 00 00 00 00  	v.nop
   22564: 00 00 00 00 00 00    	m.nop
   2256a: b1 5a 44 00  	l.ld.w r10, [r11, 0x4]
   2256e: 01 00        	c.nop
   22570: 00 00 00 00  	v.nop
   22574: 00 00 00 00 00 00    	m.nop
   2257a: f1 5a 48 00  	l.ld.w r11, [r11, 0x8]
   2257e: 01 00        	c.nop
   22580: 00 00 00 00  	v.nop
   22584: 00 00 00 00 00 00    	m.nop
   2258a: 63 52 09 19  	<unknown>
   2258e: 01 00        	c.nop
   22590: 00 00 00 00  	v.nop
   22594: 00 00 00 00 00 00    	m.nop
   2259a: 63 4a 0b 19  	<unknown>
   2259e: 01 00        	c.nop
   225a0: 00 00 00 00  	v.nop
   225a4: 00 00 00 00 00 00    	m.nop
   225aa: 2b 28 b9 01  	l.jal r0, 0x395 <0x25ef0>
   225ae: 01 00        	c.nop
   225b0: 00 00 00 00  	v.nop
   225b4: 00 00 00 00 00 00    	m.nop
   225ba: 00 00        	s.nop16
   225bc: 00 00        	s.nop16
   225be: 81 42        	<unknown>
   225c0: 00 00 00 00  	v.nop
   225c4: 00 00 00 00 00 00    	m.nop
   225ca: 71 32 00 06  	l.ld.b r9, [r6, 0x60]
   225ce: 01 00        	c.nop
   225d0: 00 00 00 00  	v.nop
   225d4: 00 00 00 00 00 00    	m.nop
   225da: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   225de: 01 00        	c.nop
   225e0: 00 00 00 00  	v.nop
   225e4: 00 00 00 00 00 00    	m.nop
   225ea: bb 4d 40 7f  	<unknown>
   225ee: 01 00        	c.nop
   225f0: 00 00 00 00  	v.nop
   225f4: 00 00 00 00 00 00    	m.nop
   225fa: 2b 78 fe 3f  	l.jal r0, -0x11 <0x224e0>
   225fe: 01 00        	c.nop
   22600: 00 00 00 00  	v.nop
   22604: 00 00 00 00 00 00    	m.nop
   2260a: bb 52 00 00  	<unknown>
   2260e: 01 00        	c.nop
   22610: 00 00 00 00  	v.nop
   22614: 00 00 00 00 00 00    	m.nop
   2261a: 31 23 40 00  	l.ld.w r12, [r4]
   2261e: 01 00        	c.nop
   22620: 00 00 00 00  	v.nop
   22624: 00 00 00 00 00 00    	m.nop
   2262a: b1 32 5c 00  	l.ld.w r10, [r6, 0x1c]
   2262e: 01 00        	c.nop
   22630: 00 00 00 00  	v.nop
   22634: 00 00 00 00 00 00    	m.nop
   2263a: 2f 58 08 29  	l.ldi16.s r11, 0x4100
   2263e: 01 00        	c.nop
   22640: 00 00 00 00  	v.nop
   22644: 00 00 00 00 00 00    	m.nop
   2264a: af 5f 91 08  	l.ldi_hi r11, 0x423e
   2264e: 01 00        	c.nop
   22650: 00 00 00 00  	v.nop
   22654: 00 00 00 00 00 00    	m.nop
   2265a: fb 60 0f 00  	<unknown>
   2265e: 01 00        	c.nop
   22660: 00 00 00 00  	v.nop
   22664: 00 00 00 00 00 00    	m.nop
   2266a: 2f 58 08 15  	l.ldi16.u r11, 0xa100
   2266e: 01 00        	c.nop
   22670: 00 00 00 00  	v.nop
   22674: 00 00 00 00 00 00    	m.nop
   2267a: 2f 58 80 08  	l.ldi_hi r11, 0x4000
   2267e: 01 00        	c.nop
   22680: 00 00 00 00  	v.nop
   22684: 00 00 00 00 00 00    	m.nop
   2268a: 31 58 ca 00  	l.st.w r10, [r11]
   2268e: 01 00        	c.nop
   22690: 00 00 00 00  	v.nop
   22694: 00 00 00 00 00 00    	m.nop
   2269a: 2b 30 81 01  	l.jal r0, 0x16 <0x227f0>
   2269e: 01 00        	c.nop
   226a0: 00 00 00 00  	v.nop
   226a4: 00 00 00 00 00 00    	m.nop
   226aa: b1 32 5c 02  	l.ld.w r10, [r6, 0x3c]
   226ae: 01 00        	c.nop
   226b0: 00 00 00 00  	v.nop
   226b4: 00 00 00 00 00 00    	m.nop
   226ba: 31 33 54 02  	l.ld.w r12, [r6, 0x34]
   226be: 01 00        	c.nop
   226c0: 00 00 00 00  	v.nop
   226c4: 00 00 00 00 00 00    	m.nop
   226ca: f1 32 50 02  	l.ld.w r11, [r6, 0x30]
   226ce: 01 00        	c.nop
   226d0: 00 00 00 00  	v.nop
   226d4: 00 00 00 00 00 00    	m.nop
   226da: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   226de: 01 00        	c.nop
   226e0: 00 00 00 00  	v.nop
   226e4: 00 00 00 00 00 00    	m.nop
   226ea: ef 77 83 00  	l.ldi_hi r14, 0x7f
   226ee: 01 00        	c.nop
   226f0: 00 00 00 00  	v.nop
   226f4: 00 00 00 00 00 00    	m.nop
   226fa: 2f 70 00 25  	l.ldi16.s r14, 0x2000
   226fe: 3a 88        	<unknown>
   22700: 00 00 00 00  	v.nop
   22704: 00 00 00 00 00 00    	m.nop
   2270a: 71 33 5c 00  	l.ld.w r13, [r6, 0x1c]
   2270e: 12 06        	<unknown>
   22710: 00 00 00 00  	v.nop
   22714: 00 00 00 00 00 00    	m.nop
   2271a: 6f 77 91 08  	l.ldi_hi r14, 0x423d
   2271e: e6 05        	<unknown>
   22720: 00 00 00 00  	v.nop
   22724: 00 00 00 00 00 00    	m.nop
   2272a: 2f 76 33 2b  	l.ldi16.s r14, 0x5678
   2272e: 3a 96        	<unknown>
   22730: 00 00 00 00  	v.nop
   22734: 00 00 00 00 00 00    	m.nop
   2273a: 2f 75 91 02  	l.ldi_hi r14, 0x1234
   2273e: 01 00        	c.nop
   22740: 00 00 00 00  	v.nop
   22744: 00 00 00 00 00 00    	m.nop
   2274a: a5 52 90 01  	l.andix r10, r10, 0x10
   2274e: 01 00        	c.nop
   22750: 00 00 00 00  	v.nop
   22754: 00 00 00 00 00 00    	m.nop
   2275a: 31 60 ce 00  	l.st.w r14, [r12]
   2275e: 01 00        	c.nop
   22760: 00 00 00 00  	v.nop
   22764: 00 00 00 00 00 00    	m.nop
   2276a: a5 5b 10 02  	l.orix r14, r11, 0x10
   2276e: 01 00        	c.nop
   22770: 00 00 00 00  	v.nop
   22774: 00 00 00 00 00 00    	m.nop
   2277a: ef 87 7f 1f  	l.ldi16.u r16, 0xffff
   2277e: 01 00        	c.nop
   22780: 00 00 00 00  	v.nop
   22784: 00 00 00 00 00 00    	m.nop
   2278a: ef 87 ff 00  	l.ldi_hi r16, 0xfff
   2278e: 01 00        	c.nop
   22790: 00 00 00 00  	v.nop
   22794: 00 00 00 00 00 00    	m.nop
   2279a: a5 5a 0a 02  	l.orix r10, r11, 0xa
   2279e: 01 00        	c.nop
   227a0: 00 00 00 00  	v.nop
   227a4: 00 00 00 00 00 00    	m.nop
   227aa: 31 61 d0 00  	l.st.w r16, [r12, 0x4]
   227ae: 01 00        	c.nop
   227b0: 00 00 00 00  	v.nop
   227b4: 00 00 00 00 00 00    	m.nop
   227ba: 31 62 ce 00  	l.st.w r14, [r12, 0x8]
   227be: 01 00        	c.nop
   227c0: 00 00 00 00  	v.nop
   227c4: 00 00 00 00 00 00    	m.nop
   227ca: 31 60 cd 00  	l.st.w r13, [r12]
   227ce: 01 00        	c.nop
   227d0: 00 00 00 00  	v.nop
   227d4: 00 00 00 00 00 00    	m.nop
   227da: 31 61 c5 00  	l.st.w r5, [r12, 0x4]
   227de: 01 00        	c.nop
   227e0: 00 00 00 00  	v.nop
   227e4: 00 00 00 00 00 00    	m.nop
   227ea: 31 62 ca 00  	l.st.w r10, [r12, 0x8]
   227ee: 01 00        	c.nop
   227f0: 00 00 00 00  	v.nop
   227f4: 00 00 00 00 00 00    	m.nop
   227fa: b1 4a 40 00  	l.ld.w r10, [r9]
   227fe: 01 00        	c.nop
   22800: 00 00 00 00  	v.nop
   22804: 00 00 00 00 00 00    	m.nop
   2280a: 00 00        	s.nop16
   2280c: 00 00        	s.nop16
   2280e: 05 05        	<unknown>
   22810: 00 00 00 00  	v.nop
   22814: 00 00 00 00 00 00    	m.nop
   2281a: 31 48 ca 00  	l.st.w r10, [r9]
   2281e: 01 00        	c.nop
   22820: 00 00 00 00  	v.nop
   22824: 00 00 00 00 00 00    	m.nop
   2282a: b1 32 0c 06  	l.ld.b r10, [r6, 0x6c]
   2282e: 01 00        	c.nop
   22830: 00 00 00 00  	v.nop
   22834: 00 00 00 00 00 00    	m.nop
   2283a: 61 3a 98 00  	l.addi.u r9, r7, 0x98
   2283e: 01 00        	c.nop
   22840: 00 00 00 00  	v.nop
   22844: 00 00 00 00 00 00    	m.nop
   2284a: a5 52 9c 01  	l.andix r10, r10, 0x1c
   2284e: 01 00        	c.nop
   22850: 00 00 00 00  	v.nop
   22854: 00 00 00 00 00 00    	m.nop
   2285a: 7b 53 20 00  	<unknown>
   2285e: 01 00        	c.nop
   22860: 00 00 00 00  	v.nop
   22864: 00 00 00 00 00 00    	m.nop
   2286a: f1 32 48 06  	l.ld.w r11, [r6, 0x68]
   2286e: 01 00        	c.nop
   22870: 00 00 00 00  	v.nop
   22874: 00 00 00 00 00 00    	m.nop
   2287a: 31 33 44 06  	l.ld.w r12, [r6, 0x64]
   2287e: 01 00        	c.nop
   22880: 00 00 00 00  	v.nop
   22884: 00 00 00 00 00 00    	m.nop
   2288a: a1 32 68 00  	l.addi.u r10, r6, 0x68
   2288e: 01 00        	c.nop
   22890: 00 00 00 00  	v.nop
   22894: 00 00 00 00 00 00    	m.nop
   2289a: fb 59 0c 40  	<unknown>
   2289e: 01 00        	c.nop
   228a0: 00 00 00 00  	v.nop
   228a4: 00 00 00 00 00 00    	m.nop
   228aa: 31 31 54 04  	l.ld.w r4, [r6, 0x54]
   228ae: 01 00        	c.nop
   228b0: 00 00 00 00  	v.nop
   228b4: 00 00 00 00 00 00    	m.nop
   228ba: f1 32 50 04  	l.ld.w r11, [r6, 0x50]
   228be: 01 00        	c.nop
   228c0: 00 00 00 00  	v.nop
   228c4: 00 00 00 00 00 00    	m.nop
   228ca: 31 50 cf 00  	l.st.w r15, [r10]
   228ce: 01 00        	c.nop
   228d0: 00 00 00 00  	v.nop
   228d4: 00 00 00 00 00 00    	m.nop
   228da: 00 00        	s.nop16
   228dc: 00 00        	s.nop16
   228de: 2e 92        	<unknown>
   228e0: 00 00 00 00  	v.nop
   228e4: 00 00 00 00 00 00    	m.nop
   228ea: 31 35 c4 04  	l.st.w r4, [r6, 0x54]
   228ee: 01 00        	c.nop
   228f0: 00 00 00 00  	v.nop
   228f4: 00 00 00 00 00 00    	m.nop
   228fa: 2b 60 80 01  	l.jal r0, 0xc <0x229b0>
   228fe: 01 00        	c.nop
   22900: 00 00 00 00  	v.nop
   22904: 00 00 00 00 00 00    	m.nop
   2290a: bb 5a 00 00  	<unknown>
   2290e: 01 00        	c.nop
   22910: 00 00 00 00  	v.nop
   22914: 00 00 00 00 00 00    	m.nop
   2291a: f1 22 40 00  	l.ld.w r11, [r4]
   2291e: 01 00        	c.nop
   22920: 00 00 00 00  	v.nop
   22924: 00 00 00 00 00 00    	m.nop
   2292a: 31 31 40 02  	l.ld.w r4, [r6, 0x20]
   2292e: 01 00        	c.nop
   22930: 00 00 00 00  	v.nop
   22934: 00 00 00 00 00 00    	m.nop
   2293a: 2f 28 08 29  	l.ldi16.s r5, 0x4100
   2293e: 01 00        	c.nop
   22940: 00 00 00 00  	v.nop
   22944: 00 00 00 00 00 00    	m.nop
   2294a: af 2f 91 08  	l.ldi_hi r5, 0x423e
   2294e: 01 00        	c.nop
   22950: 00 00 00 00  	v.nop
   22954: 00 00 00 00 00 00    	m.nop
   2295a: fb 58 0f 00  	<unknown>
   2295e: 01 00        	c.nop
   22960: 00 00 00 00  	v.nop
   22964: 00 00 00 00 00 00    	m.nop
   2296a: 2f 28 08 15  	l.ldi16.u r5, 0xa100
   2296e: 01 00        	c.nop
   22970: 00 00 00 00  	v.nop
   22974: 00 00 00 00 00 00    	m.nop
   2297a: 2f 28 80 08  	l.ldi_hi r5, 0x4000
   2297e: 01 00        	c.nop
   22980: 00 00 00 00  	v.nop
   22984: 00 00 00 00 00 00    	m.nop
   2298a: 31 28 c4 00  	l.st.w r4, [r5]
   2298e: 01 00        	c.nop
   22990: 00 00 00 00  	v.nop
   22994: 00 00 00 00 00 00    	m.nop
   2299a: 2b 30 81 01  	l.jal r0, 0x16 <0x22af0>
   2299e: 01 00        	c.nop
   229a0: 00 00 00 00  	v.nop
   229a4: 00 00 00 00 00 00    	m.nop
   229aa: 31 31 54 04  	l.ld.w r4, [r6, 0x54]
   229ae: 01 00        	c.nop
   229b0: 00 00 00 00  	v.nop
   229b4: 00 00 00 00 00 00    	m.nop
   229ba: 31 33 4c 04  	l.ld.w r12, [r6, 0x4c]
   229be: 01 00        	c.nop
   229c0: 00 00 00 00  	v.nop
   229c4: 00 00 00 00 00 00    	m.nop
   229ca: f1 32 48 04  	l.ld.w r11, [r6, 0x48]
   229ce: 01 00        	c.nop
   229d0: 00 00 00 00  	v.nop
   229d4: 00 00 00 00 00 00    	m.nop
   229da: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   229de: 01 00        	c.nop
   229e0: 00 00 00 00  	v.nop
   229e4: 00 00 00 00 00 00    	m.nop
   229ea: ef 77 83 00  	l.ldi_hi r14, 0x7f
   229ee: 01 00        	c.nop
   229f0: 00 00 00 00  	v.nop
   229f4: 00 00 00 00 00 00    	m.nop
   229fa: 2f 70 00 25  	l.ldi16.s r14, 0x2000
   229fe: 3a 88        	<unknown>
   22a00: 00 00 00 00  	v.nop
   22a04: 00 00 00 00 00 00    	m.nop
   22a0a: 71 33 40 02  	l.ld.w r13, [r6, 0x20]
   22a0e: 12 06        	<unknown>
   22a10: 00 00 00 00  	v.nop
   22a14: 00 00 00 00 00 00    	m.nop
   22a1a: 6f 77 91 08  	l.ldi_hi r14, 0x423d
   22a1e: e6 05        	<unknown>
   22a20: 00 00 00 00  	v.nop
   22a24: 00 00 00 00 00 00    	m.nop
   22a2a: 2f 76 33 2b  	l.ldi16.s r14, 0x5678
   22a2e: 3a 96        	<unknown>
   22a30: 00 00 00 00  	v.nop
   22a34: 00 00 00 00 00 00    	m.nop
   22a3a: 2f 75 91 02  	l.ldi_hi r14, 0x1234
   22a3e: 01 00        	c.nop
   22a40: 00 00 00 00  	v.nop
   22a44: 00 00 00 00 00 00    	m.nop
   22a4a: 25 21 90 01  	l.andix r4, r4, 0x10
   22a4e: 01 00        	c.nop
   22a50: 00 00 00 00  	v.nop
   22a54: 00 00 00 00 00 00    	m.nop
   22a5a: 31 60 ce 00  	l.st.w r14, [r12]
   22a5e: 01 00        	c.nop
   22a60: 00 00 00 00  	v.nop
   22a64: 00 00 00 00 00 00    	m.nop
   22a6a: a5 5b 10 02  	l.orix r14, r11, 0x10
   22a6e: 01 00        	c.nop
   22a70: 00 00 00 00  	v.nop
   22a74: 00 00 00 00 00 00    	m.nop
   22a7a: ef 87 7f 1f  	l.ldi16.u r16, 0xffff
   22a7e: 01 00        	c.nop
   22a80: 00 00 00 00  	v.nop
   22a84: 00 00 00 00 00 00    	m.nop
   22a8a: ef 87 ff 00  	l.ldi_hi r16, 0xfff
   22a8e: 01 00        	c.nop
   22a90: 00 00 00 00  	v.nop
   22a94: 00 00 00 00 00 00    	m.nop
   22a9a: 25 59 04 02  	l.orix r4, r11, 0x4
   22a9e: 01 00        	c.nop
   22aa0: 00 00 00 00  	v.nop
   22aa4: 00 00 00 00 00 00    	m.nop
   22aaa: 31 61 d0 00  	l.st.w r16, [r12, 0x4]
   22aae: 01 00        	c.nop
   22ab0: 00 00 00 00  	v.nop
   22ab4: 00 00 00 00 00 00    	m.nop
   22aba: 31 62 ce 00  	l.st.w r14, [r12, 0x8]
   22abe: 01 00        	c.nop
   22ac0: 00 00 00 00  	v.nop
   22ac4: 00 00 00 00 00 00    	m.nop
   22aca: 31 60 cd 00  	l.st.w r13, [r12]
   22ace: 01 00        	c.nop
   22ad0: 00 00 00 00  	v.nop
   22ad4: 00 00 00 00 00 00    	m.nop
   22ada: 31 61 c5 00  	l.st.w r5, [r12, 0x4]
   22ade: 01 00        	c.nop
   22ae0: 00 00 00 00  	v.nop
   22ae4: 00 00 00 00 00 00    	m.nop
   22aea: 31 62 c4 00  	l.st.w r4, [r12, 0x8]
   22aee: 01 00        	c.nop
   22af0: 00 00 00 00  	v.nop
   22af4: 00 00 00 00 00 00    	m.nop
   22afa: 31 51 40 00  	l.ld.w r4, [r10]
   22afe: 01 00        	c.nop
   22b00: 00 00 00 00  	v.nop
   22b04: 00 00 00 00 00 00    	m.nop
   22b0a: 00 00        	s.nop16
   22b0c: 00 00        	s.nop16
   22b0e: 05 02        	<unknown>
   22b10: 00 00 00 00  	v.nop
   22b14: 00 00 00 00 00 00    	m.nop
   22b1a: 31 50 c4 00  	l.st.w r4, [r10]
   22b1e: 01 00        	c.nop
   22b20: 00 00 00 00  	v.nop
   22b24: 00 00 00 00 00 00    	m.nop
   22b2a: 2f 25 01 21  	l.ldi16.s r4, 0x34
   22b2e: 01 00        	c.nop
   22b30: 00 00 00 00  	v.nop
   22b34: 00 00 00 00 00 00    	m.nop
   22b3a: 23 41 04 10  	<unknown>
   22b3e: 01 00        	c.nop
   22b40: 00 00 00 00  	v.nop
   22b44: 00 00 00 00 00 00    	m.nop
   22b4a: 31 20 da 00  	l.st.w r26, [r4]
   22b4e: 01 00        	c.nop
   22b50: 00 00 00 00  	v.nop
   22b54: 00 00 00 00 00 00    	m.nop
   22b5a: b1 3a 50 00  	l.ld.w r10, [r7, 0x10]
   22b5e: 01 00        	c.nop
   22b60: 00 00 00 00  	v.nop
   22b64: 00 00 00 00 00 00    	m.nop
   22b6a: 31 4a 40 00  	l.ld.w r8, [r9]
   22b6e: 01 00        	c.nop
   22b70: 00 00 00 00  	v.nop
   22b74: 00 00 00 00 00 00    	m.nop
   22b7a: 63 99 03 03  	l.slli r5, r19, 0x3
   22b7e: 01 00        	c.nop
   22b80: 00 00 00 00  	v.nop
   22b84: 00 00 00 00 00 00    	m.nop
   22b8a: 23 89 05 18  	<unknown>
   22b8e: 01 00        	c.nop
   22b90: 00 00 00 00  	v.nop
   22b94: 00 00 00 00 00 00    	m.nop
   22b9a: 31 21 cf 00  	l.st.w r15, [r4, 0x4]
   22b9e: 01 00        	c.nop
   22ba0: 00 00 00 00  	v.nop
   22ba4: 00 00 00 00 00 00    	m.nop
   22baa: 31 20 c6 00  	l.st.w r6, [r4]
   22bae: 01 00        	c.nop
   22bb0: 00 00 00 00  	v.nop
   22bb4: 00 00 00 00 00 00    	m.nop
   22bba: 21 39 10 00  	l.addi.u r4, r7, 0x10
   22bbe: 01 00        	c.nop
   22bc0: 00 00 00 00  	v.nop
   22bc4: 00 00 00 00 00 00    	m.nop
   22bca: 7b 51 3a 00  	<unknown>
   22bce: 01 00        	c.nop
   22bd0: 00 00 00 00  	v.nop
   22bd4: 00 00 00 00 00 00    	m.nop
   22bda: fb 54 00 40  	<unknown>
   22bde: 01 00        	c.nop
   22be0: 00 00 00 00  	v.nop
   22be4: 00 00 00 00 00 00    	m.nop
   22bea: b3 b9 83 00  	l.movs r6, md23_wrk
   22bee: 01 00        	c.nop
   22bf0: 00 00 00 00  	v.nop
   22bf4: 00 00 00 00 00 00    	m.nop
   22bfa: a5 31 96 01  	l.andix r6, r6, 0x16
   22bfe: 01 00        	c.nop
   22c00: 00 00 00 00  	v.nop
   22c04: 00 00 00 00 00 00    	m.nop
   22c0a: fb 33 00 00  	<unknown>
   22c0e: 01 00        	c.nop
   22c10: 00 00 00 00  	v.nop
   22c14: 00 00 00 00 00 00    	m.nop
   22c1a: 2f 32 04 21  	l.ldi16.s r6, 0x88
   22c1e: 01 00        	c.nop
   22c20: 00 00 00 00  	v.nop
   22c24: 00 00 00 00 00 00    	m.nop
   22c2a: a3 41 06 10  	<unknown>
   22c2e: 01 00        	c.nop
   22c30: 00 00 00 00  	v.nop
   22c34: 00 00 00 00 00 00    	m.nop
   22c3a: 71 32 40 00  	l.ld.w r9, [r6]
   22c3e: 01 00        	c.nop
   22c40: 00 00 00 00  	v.nop
   22c44: 00 00 00 00 00 00    	m.nop
   22c4a: b3 a2 83 00  	l.movs r10, md20_wrk
   22c4e: 01 00        	c.nop
   22c50: 00 00 00 00  	v.nop
   22c54: 00 00 00 00 00 00    	m.nop
   22c5a: 2f 60 f0 3e  	l.ldi_hi.m1 r12, 0xfe00
   22c5e: 06 05        	<unknown>
   22c60: 00 00 00 00  	v.nop
   22c64: 00 00 00 00 00 00    	m.nop
   22c6a: ef 67 7f 1f  	l.ldi16.u r12, 0xffff
   22c6e: 71 8d        	<unknown>
   22c70: 00 00 00 00  	v.nop
   22c74: 00 00 00 00 00 00    	m.nop
   22c7a: f3 02 80 00  	l.movs r11, sip_id
   22c7e: 01 00        	c.nop
   22c80: 00 00 00 00  	v.nop
   22c84: 00 00 00 00 00 00    	m.nop
   22c8a: 6f 60 80 00  	l.ldi_hi r12, 0x1
   22c8e: be 05        	<unknown>
   22c90: 00 00 00 00  	v.nop
   22c94: 00 00 00 00 00 00    	m.nop
   22c9a: 31 13 4c 02  	l.ld.w r12, [sp, 0x2c]
   22c9e: f1 8c        	<unknown>
   22ca0: 00 00 00 00  	v.nop
   22ca4: 00 00 00 00 00 00    	m.nop
   22caa: 00 00        	s.nop16
   22cac: 00 00        	s.nop16
   22cae: c9 8c        	<unknown>
   22cb0: 00 00 00 00  	v.nop
   22cb4: 00 00 00 00 00 00    	m.nop
   22cba: 00 00        	s.nop16
   22cbc: 00 00        	s.nop16
   22cbe: f1 8d        	<unknown>
   22cc0: 00 00 00 00  	v.nop
   22cc4: 00 00 00 00 00 00    	m.nop
   22cca: 00 00        	s.nop16
   22ccc: 00 00        	s.nop16
   22cce: cd 8c        	<unknown>
   22cd0: 00 00 00 00  	v.nop
   22cd4: 00 00 00 00 00 00    	m.nop
   22cda: 65 4a 1d 02  	l.orix r9, r9, 0x1d
   22cde: 01 00        	c.nop
   22ce0: 00 00 00 00  	v.nop
   22ce4: 00 00 00 00 00 00    	m.nop
   22cea: 31 30 c9 00  	l.st.w r9, [r6]
   22cee: 01 00        	c.nop
   22cf0: 00 00 00 00  	v.nop
   22cf4: 00 00 00 00 00 00    	m.nop
   22cfa: b1 22 40 00  	l.ld.w r10, [r4]
   22cfe: 01 00        	c.nop
   22d00: 00 00 00 00  	v.nop
   22d04: 00 00 00 00 00 00    	m.nop
   22d0a: 7b 55 1a 00  	<unknown>
   22d0e: 01 00        	c.nop
   22d10: 00 00 00 00  	v.nop
   22d14: 00 00 00 00 00 00    	m.nop
   22d1a: bb 52 20 40  	<unknown>
   22d1e: 01 00        	c.nop
   22d20: 00 00 00 00  	v.nop
   22d24: 00 00 00 00 00 00    	m.nop
   22d2a: 2f 30 08 29  	l.ldi16.s r6, 0x4100
   22d2e: 01 00        	c.nop
   22d30: 00 00 00 00  	v.nop
   22d34: 00 00 00 00 00 00    	m.nop
   22d3a: af 37 91 00  	l.ldi_hi r6, 0x23e
   22d3e: 01 00        	c.nop
   22d40: 00 00 00 00  	v.nop
   22d44: 00 00 00 00 00 00    	m.nop
   22d4a: 71 3a 00 06  	l.ld.b r9, [r7, 0x60]
   22d4e: 01 00        	c.nop
   22d50: 00 00 00 00  	v.nop
   22d54: 00 00 00 00 00 00    	m.nop
   22d5a: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   22d5e: 01 00        	c.nop
   22d60: 00 00 00 00  	v.nop
   22d64: 00 00 00 00 00 00    	m.nop
   22d6a: 3b 4e 00 00  	<unknown>
   22d6e: 01 00        	c.nop
   22d70: 00 00 00 00  	v.nop
   22d74: 00 00 00 00 00 00    	m.nop
   22d7a: b1 3a 5c 04  	l.ld.w r10, [r7, 0x5c]
   22d7e: 01 00        	c.nop
   22d80: 00 00 00 00  	v.nop
   22d84: 00 00 00 00 00 00    	m.nop
   22d8a: f1 3a 58 04  	l.ld.w r11, [r7, 0x58]
   22d8e: 01 00        	c.nop
   22d90: 00 00 00 00  	v.nop
   22d94: 00 00 00 00 00 00    	m.nop
   22d9a: 61 3a 5c 00  	l.addi.u r9, r7, 0x5c
   22d9e: 01 00        	c.nop
   22da0: 00 00 00 00  	v.nop
   22da4: 00 00 00 00 00 00    	m.nop
   22daa: bb 51 2b 40  	<unknown>
   22dae: 01 00        	c.nop
   22db0: 00 00 00 00  	v.nop
   22db4: 00 00 00 00 00 00    	m.nop
   22dba: b1 3a 5c 02  	l.ld.w r10, [r7, 0x3c]
   22dbe: 01 00        	c.nop
   22dc0: 00 00 00 00  	v.nop
   22dc4: 00 00 00 00 00 00    	m.nop
   22dca: f1 3a 58 02  	l.ld.w r11, [r7, 0x38]
   22dce: 01 00        	c.nop
   22dd0: 00 00 00 00  	v.nop
   22dd4: 00 00 00 00 00 00    	m.nop
   22dda: 31 48 cf 00  	l.st.w r15, [r9]
   22dde: 01 00        	c.nop
   22de0: 00 00 00 00  	v.nop
   22de4: 00 00 00 00 00 00    	m.nop
   22dea: 00 00        	s.nop16
   22dec: 00 00        	s.nop16
   22dee: 2e 95        	<unknown>
   22df0: 00 00 00 00  	v.nop
   22df4: 00 00 00 00 00 00    	m.nop
   22dfa: 31 3f ca 02  	l.st.w r10, [r7, 0x3c]
   22dfe: 01 00        	c.nop
   22e00: 00 00 00 00  	v.nop
   22e04: 00 00 00 00 00 00    	m.nop
   22e0a: 2b 58 82 01  	l.jal r0, 0x2b <0x230b0>
   22e0e: 01 00        	c.nop
   22e10: 00 00 00 00  	v.nop
   22e14: 00 00 00 00 00 00    	m.nop
   22e1a: b3 b9 83 00  	l.movs r6, md23_wrk
   22e1e: 91 44        	<unknown>
   22e20: 00 00 00 00  	v.nop
   22e24: 00 00 00 00 00 00    	m.nop
   22e2a: a5 31 89 01  	l.andix r6, r6, 0x9
   22e2e: 01 00        	c.nop
   22e30: 00 00 00 00  	v.nop
   22e34: 00 00 00 00 00 00    	m.nop
   22e3a: bb 37 40 7f  	<unknown>
   22e3e: 01 00        	c.nop
   22e40: 00 00 00 00  	v.nop
   22e44: 00 00 00 00 00 00    	m.nop
   22e4a: 2b 58 fe 3f  	l.jal r0, -0x15 <0x22cf0>
   22e4e: 01 00        	c.nop
   22e50: 00 00 00 00  	v.nop
   22e54: 00 00 00 00 00 00    	m.nop
   22e5a: b3 01 80 00  	l.movs r6, sip_id
   22e5e: 01 00        	c.nop
   22e60: 00 00 00 00  	v.nop
   22e64: 00 00 00 00 00 00    	m.nop
   22e6a: ef 4f 80 20  	l.ldi_hi.m1 r9, 0x1f
   22e6e: 42 03        	<unknown>
   22e70: 00 00 00 00  	v.nop
   22e74: 00 00 00 00 00 00    	m.nop
   22e7a: a5 31 89 01  	l.andix r6, r6, 0x9
   22e7e: 01 00        	c.nop
   22e80: 00 00 00 00  	v.nop
   22e84: 00 00 00 00 00 00    	m.nop
   22e8a: 2f 48 08 15  	l.ldi16.u r9, 0xa100
   22e8e: 01 00        	c.nop
   22e90: 00 00 00 00  	v.nop
   22e94: 00 00 00 00 00 00    	m.nop
   22e9a: 2f 48 90 00  	l.ldi_hi r9, 0x200
   22e9e: 01 00        	c.nop
   22ea0: 00 00 00 00  	v.nop
   22ea4: 00 00 00 00 00 00    	m.nop
   22eaa: a5 31 09 02  	l.orix r6, r6, 0x9
   22eae: 01 00        	c.nop
   22eb0: 00 00 00 00  	v.nop
   22eb4: 00 00 00 00 00 00    	m.nop
   22eba: 71 3a 00 06  	l.ld.b r9, [r7, 0x60]
   22ebe: 01 00        	c.nop
   22ec0: 00 00 00 00  	v.nop
   22ec4: 00 00 00 00 00 00    	m.nop
   22eca: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   22ece: 01 00        	c.nop
   22ed0: 00 00 00 00  	v.nop
   22ed4: 00 00 00 00 00 00    	m.nop
   22eda: bb 4a 60 7f  	<unknown>
   22ede: 01 00        	c.nop
   22ee0: 00 00 00 00  	v.nop
   22ee4: 00 00 00 00 00 00    	m.nop
   22eea: 71 3a 00 04  	l.ld.b r9, [r7, 0x40]
   22eee: 01 00        	c.nop
   22ef0: 00 00 00 00  	v.nop
   22ef4: 00 00 00 00 00 00    	m.nop
   22efa: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   22efe: 01 00        	c.nop
   22f00: 00 00 00 00  	v.nop
   22f04: 00 00 00 00 00 00    	m.nop
   22f0a: bb 4c 20 00  	<unknown>
   22f0e: 01 00        	c.nop
   22f10: 00 00 00 00  	v.nop
   22f14: 00 00 00 00 00 00    	m.nop
   22f1a: 71 3a 09 02  	l.ld.b r9, [r7, 0x29]
   22f1e: 01 00        	c.nop
   22f20: 00 00 00 00  	v.nop
   22f24: 00 00 00 00 00 00    	m.nop
   22f2a: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   22f2e: 01 00        	c.nop
   22f30: 00 00 00 00  	v.nop
   22f34: 00 00 00 00 00 00    	m.nop
   22f3a: bb 4e 00 06  	<unknown>
   22f3e: 01 00        	c.nop
   22f40: 00 00 00 00  	v.nop
   22f44: 00 00 00 00 00 00    	m.nop
   22f4a: f1 12 40 08  	l.ld.w r11, [sp, 0x80]
   22f4e: 01 00        	c.nop
   22f50: 00 00 00 00  	v.nop
   22f54: 00 00 00 00 00 00    	m.nop
   22f5a: 71 5a 40 00  	l.ld.w r9, [r11]
   22f5e: 01 00        	c.nop
   22f60: 00 00 00 00  	v.nop
   22f64: 00 00 00 00 00 00    	m.nop
   22f6a: b1 5a 44 00  	l.ld.w r10, [r11, 0x4]
   22f6e: 01 00        	c.nop
   22f70: 00 00 00 00  	v.nop
   22f74: 00 00 00 00 00 00    	m.nop
   22f7a: f1 5a 48 00  	l.ld.w r11, [r11, 0x8]
   22f7e: 01 00        	c.nop
   22f80: 00 00 00 00  	v.nop
   22f84: 00 00 00 00 00 00    	m.nop
   22f8a: 63 52 09 19  	<unknown>
   22f8e: 01 00        	c.nop
   22f90: 00 00 00 00  	v.nop
   22f94: 00 00 00 00 00 00    	m.nop
   22f9a: 63 4a 0b 19  	<unknown>
   22f9e: 01 00        	c.nop
   22fa0: 00 00 00 00  	v.nop
   22fa4: 00 00 00 00 00 00    	m.nop
   22faa: 2b 20 b1 01  	l.jal r0, 0x314 <0x260e0>
   22fae: 01 00        	c.nop
   22fb0: 00 00 00 00  	v.nop
   22fb4: 00 00 00 00 00 00    	m.nop
   22fba: 00 00        	s.nop16
   22fbc: 00 00        	s.nop16
   22fbe: 01 43        	<unknown>
   22fc0: 00 00 00 00  	v.nop
   22fc4: 00 00 00 00 00 00    	m.nop
   22fca: 71 3a 00 06  	l.ld.b r9, [r7, 0x60]
   22fce: 01 00        	c.nop
   22fd0: 00 00 00 00  	v.nop
   22fd4: 00 00 00 00 00 00    	m.nop
   22fda: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   22fde: 01 00        	c.nop
   22fe0: 00 00 00 00  	v.nop
   22fe4: 00 00 00 00 00 00    	m.nop
   22fea: 7b 4e 40 7f  	<unknown>
   22fee: 01 00        	c.nop
   22ff0: 00 00 00 00  	v.nop
   22ff4: 00 00 00 00 00 00    	m.nop
   22ffa: 2b 78 fe 3f  	l.jal r0, -0x11 <0x22ee0>
   22ffe: 01 00        	c.nop
   23000: 00 00 00 00  	v.nop
   23004: 00 00 00 00 00 00    	m.nop
   2300a: bb 52 00 00  	<unknown>
   2300e: 01 00        	c.nop
   23010: 00 00 00 00  	v.nop
   23014: 00 00 00 00 00 00    	m.nop
   2301a: 31 23 40 00  	l.ld.w r12, [r4]
   2301e: 01 00        	c.nop
   23020: 00 00 00 00  	v.nop
   23024: 00 00 00 00 00 00    	m.nop
   2302a: b1 3a 5c 00  	l.ld.w r10, [r7, 0x1c]
   2302e: 01 00        	c.nop
   23030: 00 00 00 00  	v.nop
   23034: 00 00 00 00 00 00    	m.nop
   2303a: 2f 58 08 29  	l.ldi16.s r11, 0x4100
   2303e: 01 00        	c.nop
   23040: 00 00 00 00  	v.nop
   23044: 00 00 00 00 00 00    	m.nop
   2304a: af 5f 91 08  	l.ldi_hi r11, 0x423e
   2304e: 01 00        	c.nop
   23050: 00 00 00 00  	v.nop
   23054: 00 00 00 00 00 00    	m.nop
   2305a: fb 60 0f 00  	<unknown>
   2305e: 01 00        	c.nop
   23060: 00 00 00 00  	v.nop
   23064: 00 00 00 00 00 00    	m.nop
   2306a: 2f 58 08 15  	l.ldi16.u r11, 0xa100
   2306e: 01 00        	c.nop
   23070: 00 00 00 00  	v.nop
   23074: 00 00 00 00 00 00    	m.nop
   2307a: 2f 58 80 08  	l.ldi_hi r11, 0x4000
   2307e: 01 00        	c.nop
   23080: 00 00 00 00  	v.nop
   23084: 00 00 00 00 00 00    	m.nop
   2308a: 31 58 ca 00  	l.st.w r10, [r11]
   2308e: 01 00        	c.nop
   23090: 00 00 00 00  	v.nop
   23094: 00 00 00 00 00 00    	m.nop
   2309a: 2b 30 81 01  	l.jal r0, 0x16 <0x231f0>
   2309e: 01 00        	c.nop
   230a0: 00 00 00 00  	v.nop
   230a4: 00 00 00 00 00 00    	m.nop
   230aa: b1 3a 5c 02  	l.ld.w r10, [r7, 0x3c]
   230ae: 01 00        	c.nop
   230b0: 00 00 00 00  	v.nop
   230b4: 00 00 00 00 00 00    	m.nop
   230ba: 31 3b 54 02  	l.ld.w r12, [r7, 0x34]
   230be: 01 00        	c.nop
   230c0: 00 00 00 00  	v.nop
   230c4: 00 00 00 00 00 00    	m.nop
   230ca: f1 3a 50 02  	l.ld.w r11, [r7, 0x30]
   230ce: 01 00        	c.nop
   230d0: 00 00 00 00  	v.nop
   230d4: 00 00 00 00 00 00    	m.nop
   230da: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   230de: 01 00        	c.nop
   230e0: 00 00 00 00  	v.nop
   230e4: 00 00 00 00 00 00    	m.nop
   230ea: ef 77 83 00  	l.ldi_hi r14, 0x7f
   230ee: 01 00        	c.nop
   230f0: 00 00 00 00  	v.nop
   230f4: 00 00 00 00 00 00    	m.nop
   230fa: 2f 70 00 25  	l.ldi16.s r14, 0x2000
   230fe: 3a 88        	<unknown>
   23100: 00 00 00 00  	v.nop
   23104: 00 00 00 00 00 00    	m.nop
   2310a: 71 3b 5c 00  	l.ld.w r13, [r7, 0x1c]
   2310e: 12 06        	<unknown>
   23110: 00 00 00 00  	v.nop
   23114: 00 00 00 00 00 00    	m.nop
   2311a: 6f 77 91 08  	l.ldi_hi r14, 0x423d
   2311e: e6 05        	<unknown>
   23120: 00 00 00 00  	v.nop
   23124: 00 00 00 00 00 00    	m.nop
   2312a: 2f 76 33 2b  	l.ldi16.s r14, 0x5678
   2312e: 3a 96        	<unknown>
   23130: 00 00 00 00  	v.nop
   23134: 00 00 00 00 00 00    	m.nop
   2313a: 2f 75 91 02  	l.ldi_hi r14, 0x1234
   2313e: 01 00        	c.nop
   23140: 00 00 00 00  	v.nop
   23144: 00 00 00 00 00 00    	m.nop
   2314a: a5 52 90 01  	l.andix r10, r10, 0x10
   2314e: 01 00        	c.nop
   23150: 00 00 00 00  	v.nop
   23154: 00 00 00 00 00 00    	m.nop
   2315a: 31 60 ce 00  	l.st.w r14, [r12]
   2315e: 01 00        	c.nop
   23160: 00 00 00 00  	v.nop
   23164: 00 00 00 00 00 00    	m.nop
   2316a: a5 5b 10 02  	l.orix r14, r11, 0x10
   2316e: 01 00        	c.nop
   23170: 00 00 00 00  	v.nop
   23174: 00 00 00 00 00 00    	m.nop
   2317a: ef 87 7f 1f  	l.ldi16.u r16, 0xffff
   2317e: 01 00        	c.nop
   23180: 00 00 00 00  	v.nop
   23184: 00 00 00 00 00 00    	m.nop
   2318a: ef 87 ff 00  	l.ldi_hi r16, 0xfff
   2318e: 01 00        	c.nop
   23190: 00 00 00 00  	v.nop
   23194: 00 00 00 00 00 00    	m.nop
   2319a: a5 5a 0a 02  	l.orix r10, r11, 0xa
   2319e: 01 00        	c.nop
   231a0: 00 00 00 00  	v.nop
   231a4: 00 00 00 00 00 00    	m.nop
   231aa: 31 61 d0 00  	l.st.w r16, [r12, 0x4]
   231ae: 01 00        	c.nop
   231b0: 00 00 00 00  	v.nop
   231b4: 00 00 00 00 00 00    	m.nop
   231ba: 31 62 ce 00  	l.st.w r14, [r12, 0x8]
   231be: 01 00        	c.nop
   231c0: 00 00 00 00  	v.nop
   231c4: 00 00 00 00 00 00    	m.nop
   231ca: 31 60 cd 00  	l.st.w r13, [r12]
   231ce: 01 00        	c.nop
   231d0: 00 00 00 00  	v.nop
   231d4: 00 00 00 00 00 00    	m.nop
   231da: 31 61 c6 00  	l.st.w r6, [r12, 0x4]
   231de: 01 00        	c.nop
   231e0: 00 00 00 00  	v.nop
   231e4: 00 00 00 00 00 00    	m.nop
   231ea: 31 62 ca 00  	l.st.w r10, [r12, 0x8]
   231ee: 01 00        	c.nop
   231f0: 00 00 00 00  	v.nop
   231f4: 00 00 00 00 00 00    	m.nop
   231fa: b1 4a 40 00  	l.ld.w r10, [r9]
   231fe: 01 00        	c.nop
   23200: 00 00 00 00  	v.nop
   23204: 00 00 00 00 00 00    	m.nop
   2320a: 00 00        	s.nop16
   2320c: 00 00        	s.nop16
   2320e: 05 05        	<unknown>
   23210: 00 00 00 00  	v.nop
   23214: 00 00 00 00 00 00    	m.nop
   2321a: 31 48 ca 00  	l.st.w r10, [r9]
   2321e: 01 00        	c.nop
   23220: 00 00 00 00  	v.nop
   23224: 00 00 00 00 00 00    	m.nop
   2322a: 71 3a 0c 06  	l.ld.b r9, [r7, 0x6c]
   2322e: 01 00        	c.nop
   23230: 00 00 00 00  	v.nop
   23234: 00 00 00 00 00 00    	m.nop
   2323a: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   2323e: 01 00        	c.nop
   23240: 00 00 00 00  	v.nop
   23244: 00 00 00 00 00 00    	m.nop
   2324a: 7b 4b 20 00  	<unknown>
   2324e: 01 00        	c.nop
   23250: 00 00 00 00  	v.nop
   23254: 00 00 00 00 00 00    	m.nop
   2325a: b1 3a 48 06  	l.ld.w r10, [r7, 0x68]
   2325e: 01 00        	c.nop
   23260: 00 00 00 00  	v.nop
   23264: 00 00 00 00 00 00    	m.nop
   2326a: f1 3a 44 06  	l.ld.w r11, [r7, 0x64]
   2326e: 01 00        	c.nop
   23270: 00 00 00 00  	v.nop
   23274: 00 00 00 00 00 00    	m.nop
   2327a: 61 3a 68 00  	l.addi.u r9, r7, 0x68
   2327e: 01 00        	c.nop
   23280: 00 00 00 00  	v.nop
   23284: 00 00 00 00 00 00    	m.nop
   2328a: fb 51 0b 40  	<unknown>
   2328e: 01 00        	c.nop
   23290: 00 00 00 00  	v.nop
   23294: 00 00 00 00 00 00    	m.nop
   2329a: 31 39 54 04  	l.ld.w r4, [r7, 0x54]
   2329e: 01 00        	c.nop
   232a0: 00 00 00 00  	v.nop
   232a4: 00 00 00 00 00 00    	m.nop
   232aa: b1 3a 50 04  	l.ld.w r10, [r7, 0x50]
   232ae: 01 00        	c.nop
   232b0: 00 00 00 00  	v.nop
   232b4: 00 00 00 00 00 00    	m.nop
   232ba: 31 48 cf 00  	l.st.w r15, [r9]
   232be: 01 00        	c.nop
   232c0: 00 00 00 00  	v.nop
   232c4: 00 00 00 00 00 00    	m.nop
   232ca: 00 00        	s.nop16
   232cc: 00 00        	s.nop16
   232ce: 2a 92        	<unknown>
   232d0: 00 00 00 00  	v.nop
   232d4: 00 00 00 00 00 00    	m.nop
   232da: 31 3d c4 04  	l.st.w r4, [r7, 0x54]
   232de: 01 00        	c.nop
   232e0: 00 00 00 00  	v.nop
   232e4: 00 00 00 00 00 00    	m.nop
   232ea: 2b 60 80 01  	l.jal r0, 0xc <0x233a0>
   232ee: 01 00        	c.nop
   232f0: 00 00 00 00  	v.nop
   232f4: 00 00 00 00 00 00    	m.nop
   232fa: bb 52 00 00  	<unknown>
   232fe: 01 00        	c.nop
   23300: 00 00 00 00  	v.nop
   23304: 00 00 00 00 00 00    	m.nop
   2330a: b1 22 40 00  	l.ld.w r10, [r4]
   2330e: 01 00        	c.nop
   23310: 00 00 00 00  	v.nop
   23314: 00 00 00 00 00 00    	m.nop
   2331a: 31 39 40 02  	l.ld.w r4, [r7, 0x20]
   2331e: 01 00        	c.nop
   23320: 00 00 00 00  	v.nop
   23324: 00 00 00 00 00 00    	m.nop
   2332a: 2f 30 08 29  	l.ldi16.s r6, 0x4100
   2332e: 01 00        	c.nop
   23330: 00 00 00 00  	v.nop
   23334: 00 00 00 00 00 00    	m.nop
   2333a: af 37 91 08  	l.ldi_hi r6, 0x423e
   2333e: 01 00        	c.nop
   23340: 00 00 00 00  	v.nop
   23344: 00 00 00 00 00 00    	m.nop
   2334a: fb 50 0f 00  	<unknown>
   2334e: 01 00        	c.nop
   23350: 00 00 00 00  	v.nop
   23354: 00 00 00 00 00 00    	m.nop
   2335a: 2f 30 08 15  	l.ldi16.u r6, 0xa100
   2335e: 01 00        	c.nop
   23360: 00 00 00 00  	v.nop
   23364: 00 00 00 00 00 00    	m.nop
   2336a: 2f 30 80 08  	l.ldi_hi r6, 0x4000
   2336e: 01 00        	c.nop
   23370: 00 00 00 00  	v.nop
   23374: 00 00 00 00 00 00    	m.nop
   2337a: 31 30 c4 00  	l.st.w r4, [r6]
   2337e: 01 00        	c.nop
   23380: 00 00 00 00  	v.nop
   23384: 00 00 00 00 00 00    	m.nop
   2338a: 2b 30 81 01  	l.jal r0, 0x16 <0x234e0>
   2338e: 01 00        	c.nop
   23390: 00 00 00 00  	v.nop
   23394: 00 00 00 00 00 00    	m.nop
   2339a: 31 39 54 04  	l.ld.w r4, [r7, 0x54]
   2339e: 01 00        	c.nop
   233a0: 00 00 00 00  	v.nop
   233a4: 00 00 00 00 00 00    	m.nop
   233aa: f1 3a 4c 04  	l.ld.w r11, [r7, 0x4c]
   233ae: 01 00        	c.nop
   233b0: 00 00 00 00  	v.nop
   233b4: 00 00 00 00 00 00    	m.nop
   233ba: b1 3a 48 04  	l.ld.w r10, [r7, 0x48]
   233be: 01 00        	c.nop
   233c0: 00 00 00 00  	v.nop
   233c4: 00 00 00 00 00 00    	m.nop
   233ca: ef 6f 7f 1f  	l.ldi16.u r13, 0xffff
   233ce: 01 00        	c.nop
   233d0: 00 00 00 00  	v.nop
   233d4: 00 00 00 00 00 00    	m.nop
   233da: ef 6f 83 00  	l.ldi_hi r13, 0x7f
   233de: 01 00        	c.nop
   233e0: 00 00 00 00  	v.nop
   233e4: 00 00 00 00 00 00    	m.nop
   233ea: 2f 68 00 25  	l.ldi16.s r13, 0x2000
   233ee: 36 87        	<unknown>
   233f0: 00 00 00 00  	v.nop
   233f4: 00 00 00 00 00 00    	m.nop
   233fa: 31 3b 40 02  	l.ld.w r12, [r7, 0x20]
   233fe: 92 05        	<unknown>
   23400: 00 00 00 00  	v.nop
   23404: 00 00 00 00 00 00    	m.nop
   2340a: 6f 6f 91 08  	l.ldi_hi r13, 0x423d
   2340e: 66 05        	<unknown>
   23410: 00 00 00 00  	v.nop
   23414: 00 00 00 00 00 00    	m.nop
   2341a: 2f 6e 33 2b  	l.ldi16.s r13, 0x5678
   2341e: b6 95        	<unknown>
   23420: 00 00 00 00  	v.nop
   23424: 00 00 00 00 00 00    	m.nop
   2342a: 2f 6d 91 02  	l.ldi_hi r13, 0x1234
   2342e: 01 00        	c.nop
   23430: 00 00 00 00  	v.nop
   23434: 00 00 00 00 00 00    	m.nop
   2343a: 25 21 8e 01  	l.andix r4, r4, 0xe
   2343e: 01 00        	c.nop
   23440: 00 00 00 00  	v.nop
   23444: 00 00 00 00 00 00    	m.nop
   2344a: 31 58 cd 00  	l.st.w r13, [r11]
   2344e: 01 00        	c.nop
   23450: 00 00 00 00  	v.nop
   23454: 00 00 00 00 00 00    	m.nop
   2345a: 65 53 0e 02  	l.orix r13, r10, 0xe
   2345e: 01 00        	c.nop
   23460: 00 00 00 00  	v.nop
   23464: 00 00 00 00 00 00    	m.nop
   2346a: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   2346e: 01 00        	c.nop
   23470: 00 00 00 00  	v.nop
   23474: 00 00 00 00 00 00    	m.nop
   2347a: ef 77 ff 00  	l.ldi_hi r14, 0xfff
   2347e: 01 00        	c.nop
   23480: 00 00 00 00  	v.nop
   23484: 00 00 00 00 00 00    	m.nop
   2348a: 25 51 04 02  	l.orix r4, r10, 0x4
   2348e: 01 00        	c.nop
   23490: 00 00 00 00  	v.nop
   23494: 00 00 00 00 00 00    	m.nop
   2349a: 31 59 ce 00  	l.st.w r14, [r11, 0x4]
   2349e: 01 00        	c.nop
   234a0: 00 00 00 00  	v.nop
   234a4: 00 00 00 00 00 00    	m.nop
   234aa: 31 5a cd 00  	l.st.w r13, [r11, 0x8]
   234ae: 01 00        	c.nop
   234b0: 00 00 00 00  	v.nop
   234b4: 00 00 00 00 00 00    	m.nop
   234ba: 31 58 cc 00  	l.st.w r12, [r11]
   234be: 01 00        	c.nop
   234c0: 00 00 00 00  	v.nop
   234c4: 00 00 00 00 00 00    	m.nop
   234ca: 31 59 c6 00  	l.st.w r6, [r11, 0x4]
   234ce: 01 00        	c.nop
   234d0: 00 00 00 00  	v.nop
   234d4: 00 00 00 00 00 00    	m.nop
   234da: 31 5a c4 00  	l.st.w r4, [r11, 0x8]
   234de: 01 00        	c.nop
   234e0: 00 00 00 00  	v.nop
   234e4: 00 00 00 00 00 00    	m.nop
   234ea: 31 49 40 00  	l.ld.w r4, [r9]
   234ee: 01 00        	c.nop
   234f0: 00 00 00 00  	v.nop
   234f4: 00 00 00 00 00 00    	m.nop
   234fa: 00 00        	s.nop16
   234fc: 00 00        	s.nop16
   234fe: 05 02        	<unknown>
   23500: 00 00 00 00  	v.nop
   23504: 00 00 00 00 00 00    	m.nop
   2350a: 31 48 c4 00  	l.st.w r4, [r9]
   2350e: 01 00        	c.nop
   23510: 00 00 00 00  	v.nop
   23514: 00 00 00 00 00 00    	m.nop
   2351a: b1 11 54 06  	l.ld.w r6, [sp, 0x74]
   2351e: 01 00        	c.nop
   23520: 00 00 00 00  	v.nop
   23524: 00 00 00 00 00 00    	m.nop
   2352a: 2f 25 01 21  	l.ldi16.s r4, 0x34
   2352e: 01 00        	c.nop
   23530: 00 00 00 00  	v.nop
   23534: 00 00 00 00 00 00    	m.nop
   2353a: 23 41 04 10  	<unknown>
   2353e: 01 00        	c.nop
   23540: 00 00 00 00  	v.nop
   23544: 00 00 00 00 00 00    	m.nop
   2354a: 31 20 da 00  	l.st.w r26, [r4]
   2354e: 01 00        	c.nop
   23550: 00 00 00 00  	v.nop
   23554: 00 00 00 00 00 00    	m.nop
   2355a: 63 31 05 18  	<unknown>
   2355e: 01 00        	c.nop
   23560: 00 00 00 00  	v.nop
   23564: 00 00 00 00 00 00    	m.nop
   2356a: 31 29 cf 00  	l.st.w r15, [r5, 0x4]
   2356e: 01 00        	c.nop
   23570: 00 00 00 00  	v.nop
   23574: 00 00 00 00 00 00    	m.nop
   2357a: 31 28 c7 00  	l.st.w r7, [r5]
   2357e: 01 00        	c.nop
   23580: 00 00 00 00  	v.nop
   23584: 00 00 00 00 00 00    	m.nop
   2358a: fb f0 00 40  	<unknown>
   2358e: 01 00        	c.nop
   23590: 00 00 00 00  	v.nop
   23594: 00 00 00 00 00 00    	m.nop
   2359a: 31 11 48 06  	l.ld.w r4, [sp, 0x68]
   2359e: 01 00        	c.nop
   235a0: 00 00 00 00  	v.nop
   235a4: 00 00 00 00 00 00    	m.nop
   235aa: 2d 20 20 00  	l.wait.dec r4
   235ae: 01 00        	c.nop
   235b0: 00 00 00 00  	v.nop
   235b4: 00 00 00 00 00 00    	m.nop
   235ba: e3 a4 03 03  	l.slli r19, r20, 0x3
   235be: 01 00        	c.nop
   235c0: 00 00 00 00  	v.nop
   235c4: 00 00 00 00 00 00    	m.nop
   235ca: 23 89 13 18  	<unknown>
   235ce: 01 00        	c.nop
   235d0: 00 00 00 00  	v.nop
   235d4: 00 00 00 00 00 00    	m.nop
   235da: 31 21 40 00  	l.ld.w r4, [r4]
   235de: 01 00        	c.nop
   235e0: 00 00 00 00  	v.nop
   235e4: 00 00 00 00 00 00    	m.nop
   235ea: 2f 2a 10 21  	l.ldi16.s r5, 0x208
   235ee: 01 00        	c.nop
   235f0: 00 00 00 00  	v.nop
   235f4: 00 00 00 00 00 00    	m.nop
   235fa: 00 00        	s.nop16
   235fc: 00 00        	s.nop16
   235fe: 16 92        	<unknown>
   23600: 00 00 00 00  	v.nop
   23604: 00 00 00 00 00 00    	m.nop
   2360a: 71 21 10 04  	l.ld.b r5, [r4, 0x50]
   2360e: 01 00        	c.nop
   23610: 00 00 00 00  	v.nop
   23614: 00 00 00 00 00 00    	m.nop
   2361a: 65 29 9c 01  	l.andix r5, r5, 0x1c
   2361e: 01 00        	c.nop
   23620: 00 00 00 00  	v.nop
   23624: 00 00 00 00 00 00    	m.nop
   2362a: fb 2a 00 00  	<unknown>
   2362e: 01 00        	c.nop
   23630: 00 00 00 00  	v.nop
   23634: 00 00 00 00 00 00    	m.nop
   2363a: b1 21 4c 04  	l.ld.w r6, [r4, 0x4c]
   2363e: 01 00        	c.nop
   23640: 00 00 00 00  	v.nop
   23644: 00 00 00 00 00 00    	m.nop
   2364a: f1 21 48 04  	l.ld.w r7, [r4, 0x48]
   2364e: 01 00        	c.nop
   23650: 00 00 00 00  	v.nop
   23654: 00 00 00 00 00 00    	m.nop
   2365a: 61 21 4c 00  	l.addi.u r5, r4, 0x4c
   2365e: 01 00        	c.nop
   23660: 00 00 00 00  	v.nop
   23664: 00 00 00 00 00 00    	m.nop
   2366a: 3b 33 07 40  	<unknown>
   2366e: 01 00        	c.nop
   23670: 00 00 00 00  	v.nop
   23674: 00 00 00 00 00 00    	m.nop
   2367a: f1 21 4c 02  	l.ld.w r7, [r4, 0x2c]
   2367e: 01 43        	<unknown>
   23680: 00 00 00 00  	v.nop
   23684: 00 00 00 00 00 00    	m.nop
   2368a: 31 22 48 02  	l.ld.w r8, [r4, 0x28]
   2368e: 01 00        	c.nop
   23690: 00 00 00 00  	v.nop
   23694: 00 00 00 00 00 00    	m.nop
   2369a: 31 23 c0 04  	l.st.w r0, [r4, 0x4c]
   2369e: 01 00        	c.nop
   236a0: 00 00 00 00  	v.nop
   236a4: 00 00 00 00 00 00    	m.nop
   236aa: 00 00        	s.nop16
   236ac: 00 00        	s.nop16
   236ae: a2 93        	<unknown>
   236b0: 00 00 00 00  	v.nop
   236b4: 00 00 00 00 00 00    	m.nop
   236ba: 31 23 c7 02  	l.st.w r7, [r4, 0x2c]
   236be: 01 00        	c.nop
   236c0: 00 00 00 00  	v.nop
   236c4: 00 00 00 00 00 00    	m.nop
   236ca: 2b 30 80 01  	l.jal r0, 0x6 <0x23720>
   236ce: 01 00        	c.nop
   236d0: 00 00 00 00  	v.nop
   236d4: 00 00 00 00 00 00    	m.nop
   236da: 71 21 10 02  	l.ld.b r5, [r4, 0x30]
   236de: 01 00        	c.nop
   236e0: 00 00 00 00  	v.nop
   236e4: 00 00 00 00 00 00    	m.nop
   236ea: 65 29 9c 01  	l.andix r5, r5, 0x1c
   236ee: 01 00        	c.nop
   236f0: 00 00 00 00  	v.nop
   236f4: 00 00 00 00 00 00    	m.nop
   236fa: bb 28 20 00  	<unknown>
   236fe: 01 00        	c.nop
   23700: 00 00 00 00  	v.nop
   23704: 00 00 00 00 00 00    	m.nop
   2370a: 61 21 2c 00  	l.addi.u r5, r4, 0x2c
   2370e: 01 00        	c.nop
   23710: 00 00 00 00  	v.nop
   23714: 00 00 00 00 00 00    	m.nop
   2371a: b1 21 4c 02  	l.ld.w r6, [r4, 0x2c]
   2371e: 01 00        	c.nop
   23720: 00 00 00 00  	v.nop
   23724: 00 00 00 00 00 00    	m.nop
   2372a: 00 00        	s.nop16
   2372c: 00 00        	s.nop16
   2372e: 05 03        	<unknown>
   23730: 00 00 00 00  	v.nop
   23734: 00 00 00 00 00 00    	m.nop
   2373a: 31 28 c6 00  	l.st.w r6, [r5]
   2373e: 01 00        	c.nop
   23740: 00 00 00 00  	v.nop
   23744: 00 00 00 00 00 00    	m.nop
   2374a: 73 01 80 00  	l.movs r5, sip_id
   2374e: 01 00        	c.nop
   23750: 00 00 00 00  	v.nop
   23754: 00 00 00 00 00 00    	m.nop
   2375a: b3 79 83 00  	l.movs r6, md15_wrk
   2375e: c2 02        	<unknown>
   23760: 00 00 00 00  	v.nop
   23764: 00 00 00 00 00 00    	m.nop
   2376a: a3 31 90 03  	l.srli r6, r6, 0x10
   2376e: 1f f0        	<unknown>
   23770: 00 00 00 00  	v.nop
   23774: 00 00 00 00 00 00    	m.nop
   2377a: a5 31 9c 01  	l.andix r6, r6, 0x1c
   2377e: 01 00        	c.nop
   23780: 00 00 00 00  	v.nop
   23784: 00 00 00 00 00 00    	m.nop
   2378a: 71 22 4c 02  	l.ld.w r9, [r4, 0x2c]
   2378e: 22 93        	<unknown>
   23790: 00 00 00 00  	v.nop
   23794: 00 00 00 00 00 00    	m.nop
   2379a: ef 47 80 20  	l.ldi_hi.m1 r8, 0x1f
   2379e: 12 03        	<unknown>
   237a0: 00 00 00 00  	v.nop
   237a4: 00 00 00 00 00 00    	m.nop
   237aa: 65 29 88 01  	l.andix r5, r5, 0x8
   237ae: 01 00        	c.nop
   237b0: 00 00 00 00  	v.nop
   237b4: 00 00 00 00 00 00    	m.nop
   237ba: ef 47 7f 1f  	l.ldi16.u r8, 0xffff
   237be: 01 00        	c.nop
   237c0: 00 00 00 00  	v.nop
   237c4: 00 00 00 00 00 00    	m.nop
   237ca: ef 47 83 00  	l.ldi_hi r8, 0x7f
   237ce: 01 00        	c.nop
   237d0: 00 00 00 00  	v.nop
   237d4: 00 00 00 00 00 00    	m.nop
   237da: f1 21 40 02  	l.ld.w r7, [r4, 0x20]
   237de: a2 85        	<unknown>
   237e0: 00 00 00 00  	v.nop
   237e4: 00 00 00 00 00 00    	m.nop
   237ea: 25 4a 8b 01  	l.andix r8, r9, 0xb
   237ee: 01 00        	c.nop
   237f0: 00 00 00 00  	v.nop
   237f4: 00 00 00 00 00 00    	m.nop
   237fa: 2f 48 00 25  	l.ldi16.s r9, 0x2000
   237fe: 01 00        	c.nop
   23800: 00 00 00 00  	v.nop
   23804: 00 00 00 00 00 00    	m.nop
   2380a: 6f 4f 91 08  	l.ldi_hi r9, 0x423d
   2380e: 01 00        	c.nop
   23810: 00 00 00 00  	v.nop
   23814: 00 00 00 00 00 00    	m.nop
   2381a: 2f 48 20 25  	l.ldi16.s r9, 0x2400
   2381e: 26 93        	<unknown>
   23820: 00 00 00 00  	v.nop
   23824: 00 00 00 00 00 00    	m.nop
   2382a: 2f 48 90 00  	l.ldi_hi r9, 0x200
   2382e: e6 03        	<unknown>
   23830: 00 00 00 00  	v.nop
   23834: 00 00 00 00 00 00    	m.nop
   2383a: b1 22 4c 00  	l.ld.w r10, [r4, 0xc]
   2383e: 01 00        	c.nop
   23840: 00 00 00 00  	v.nop
   23844: 00 00 00 00 00 00    	m.nop
   2384a: 65 29 09 02  	l.orix r5, r5, 0x9
   2384e: 01 00        	c.nop
   23850: 00 00 00 00  	v.nop
   23854: 00 00 00 00 00 00    	m.nop
   2385a: 2f 4e 33 2b  	l.ldi16.s r9, 0x5678
   2385e: 01 00        	c.nop
   23860: 00 00 00 00  	v.nop
   23864: 00 00 00 00 00 00    	m.nop
   2386a: 2f 4d 91 02  	l.ldi_hi r9, 0x1234
   2386e: 01 00        	c.nop
   23870: 00 00 00 00  	v.nop
   23874: 00 00 00 00 00 00    	m.nop
   2387a: 31 30 c9 00  	l.st.w r9, [r6]
   2387e: 01 00        	c.nop
   23880: 00 00 00 00  	v.nop
   23884: 00 00 00 00 00 00    	m.nop
   2388a: 65 3a 0b 02  	l.orix r9, r7, 0xb
   2388e: 01 00        	c.nop
   23890: 00 00 00 00  	v.nop
   23894: 00 00 00 00 00 00    	m.nop
   2389a: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   2389e: 01 00        	c.nop
   238a0: 00 00 00 00  	v.nop
   238a4: 00 00 00 00 00 00    	m.nop
   238aa: ef 5f ff 00  	l.ldi_hi r11, 0xfff
   238ae: 01 00        	c.nop
   238b0: 00 00 00 00  	v.nop
   238b4: 00 00 00 00 00 00    	m.nop
   238ba: e5 41 07 02  	l.orix r7, r8, 0x7
   238be: 01 00        	c.nop
   238c0: 00 00 00 00  	v.nop
   238c4: 00 00 00 00 00 00    	m.nop
   238ca: 31 31 cb 00  	l.st.w r11, [r6, 0x4]
   238ce: 01 00        	c.nop
   238d0: 00 00 00 00  	v.nop
   238d4: 00 00 00 00 00 00    	m.nop
   238da: 31 32 c9 00  	l.st.w r9, [r6, 0x8]
   238de: 01 00        	c.nop
   238e0: 00 00 00 00  	v.nop
   238e4: 00 00 00 00 00 00    	m.nop
   238ea: 31 30 ca 00  	l.st.w r10, [r6]
   238ee: 01 00        	c.nop
   238f0: 00 00 00 00  	v.nop
   238f4: 00 00 00 00 00 00    	m.nop
   238fa: 31 31 c5 00  	l.st.w r5, [r6, 0x4]
   238fe: 01 00        	c.nop
   23900: 00 00 00 00  	v.nop
   23904: 00 00 00 00 00 00    	m.nop
   2390a: 31 32 c7 00  	l.st.w r7, [r6, 0x8]
   2390e: 01 00        	c.nop
   23910: 00 00 00 00  	v.nop
   23914: 00 00 00 00 00 00    	m.nop
   2391a: 71 11 54 0c  	l.ld.w r5, [sp, 0xd4]
   2391e: 01 00        	c.nop
   23920: 00 00 00 00  	v.nop
   23924: 00 00 00 00 00 00    	m.nop
   2392a: 31 21 4c 00  	l.ld.w r4, [r4, 0xc]
   2392e: 01 00        	c.nop
   23930: 00 00 00 00  	v.nop
   23934: 00 00 00 00 00 00    	m.nop
   2393a: 2d 20 20 00  	l.wait.dec r4
   2393e: 01 00        	c.nop
   23940: 00 00 00 00  	v.nop
   23944: 00 00 00 00 00 00    	m.nop
   2394a: bb a0 0f 00  	<unknown>
   2394e: 01 00        	c.nop
   23950: 00 00 00 00  	v.nop
   23954: 00 00 00 00 00 00    	m.nop
   2395a: 71 11 4c 12  	l.ld.w r5, [sp, 0x12c]
   2395e: 01 00        	c.nop
   23960: 00 00 00 00  	v.nop
   23964: 00 00 00 00 00 00    	m.nop
   2396a: 31 11 44 16  	l.ld.w r4, [sp, 0x164]
   2396e: d2 88        	<unknown>
   23970: 00 00 00 00  	v.nop
   23974: 00 00 00 00 00 00    	m.nop
   2397a: 2f 30 70 2b  	l.ldi16.s r6, 0x5e00
   2397e: 01 00        	c.nop
   23980: 00 00 00 00  	v.nop
   23984: 00 00 00 00 00 00    	m.nop
   2398a: ef 34 80 00  	l.ldi_hi r6, 0x13
   2398e: 01 00        	c.nop
   23990: 00 00 00 00  	v.nop
   23994: 00 00 00 00 00 00    	m.nop
   2399a: 00 00        	s.nop16
   2399c: 00 00        	s.nop16
   2399e: 1a 92        	<unknown>
   239a0: 00 00 00 00  	v.nop
   239a4: 00 00 00 00 00 00    	m.nop
   239aa: bb 20 0f 40  	<unknown>
   239ae: 01 00        	c.nop
   239b0: 00 00 00 00  	v.nop
   239b4: 00 00 00 00 00 00    	m.nop
   239ba: 2f 20 c0 20  	l.ldi_hi.m1 r4, 0x800
   239be: 01 00        	c.nop
   239c0: 00 00 00 00  	v.nop
   239c4: 00 00 00 00 00 00    	m.nop
   239ca: 2f 30 81 20  	l.ldi_hi.m1 r6, 0x20
   239ce: 01 00        	c.nop
   239d0: 00 00 00 00  	v.nop
   239d4: 00 00 00 00 00 00    	m.nop
   239da: b1 11 4c 22  	l.ld.w r6, [sp, 0x22c]
   239de: 9a 92        	<unknown>
   239e0: 00 00 00 00  	v.nop
   239e4: 00 00 00 00 00 00    	m.nop
   239ea: 2f 38 70 1b  	l.ldi16.u r7, 0xde00
   239ee: 01 00        	c.nop
   239f0: 00 00 00 00  	v.nop
   239f4: 00 00 00 00 00 00    	m.nop
   239fa: 6f 3a 81 00  	l.ldi_hi r7, 0x29
   239fe: 01 00        	c.nop
   23a00: 00 00 00 00  	v.nop
   23a04: 00 00 00 00 00 00    	m.nop
   23a0a: 2f 48 10 21  	l.ldi16.s r9, 0x200
   23a0e: 01 00        	c.nop
   23a10: 00 00 00 00  	v.nop
   23a14: 00 00 00 00 00 00    	m.nop
   23a1a: 31 20 cf 08  	l.st.w r15, [r4, 0x80]
   23a1e: 01 00        	c.nop
   23a20: 00 00 00 00  	v.nop
   23a24: 00 00 00 00 00 00    	m.nop
   23a2a: 6f 38 70 1f  	l.ldi16.u r7, 0xfe01
   23a2e: 1e 93        	<unknown>
   23a30: 00 00 00 00  	v.nop
   23a34: 00 00 00 00 00 00    	m.nop
   23a3a: 6f 38 f0 1e  	l.ldi_hi r7, 0xfe01
   23a3e: 01 00        	c.nop
   23a40: 00 00 00 00  	v.nop
   23a44: 00 00 00 00 00 00    	m.nop
   23a4a: 6f 3a 70 1f  	l.ldi16.u r7, 0xfe09
   23a4e: 1e 84        	<unknown>
   23a50: 00 00 00 00  	v.nop
   23a54: 00 00 00 00 00 00    	m.nop
   23a5a: 6f 3a f0 1e  	l.ldi_hi r7, 0xfe09
   23a5e: 01 00        	c.nop
   23a60: 00 00 00 00  	v.nop
   23a64: 00 00 00 00 00 00    	m.nop
   23a6a: 31 21 c7 00  	l.st.w r7, [r4, 0x4]
   23a6e: 01 00        	c.nop
   23a70: 00 00 00 00  	v.nop
   23a74: 00 00 00 00 00 00    	m.nop
   23a7a: 31 23 c7 00  	l.st.w r7, [r4, 0xc]
   23a7e: 01 00        	c.nop
   23a80: 00 00 00 00  	v.nop
   23a84: 00 00 00 00 00 00    	m.nop
   23a8a: 31 25 c7 00  	l.st.w r7, [r4, 0x14]
   23a8e: 01 00        	c.nop
   23a90: 00 00 00 00  	v.nop
   23a94: 00 00 00 00 00 00    	m.nop
   23a9a: 31 27 c7 00  	l.st.w r7, [r4, 0x1c]
   23a9e: 01 00        	c.nop
   23aa0: 00 00 00 00  	v.nop
   23aa4: 00 00 00 00 00 00    	m.nop
   23aaa: 31 21 c7 02  	l.st.w r7, [r4, 0x24]
   23aae: 01 00        	c.nop
   23ab0: 00 00 00 00  	v.nop
   23ab4: 00 00 00 00 00 00    	m.nop
   23aba: 31 23 c7 02  	l.st.w r7, [r4, 0x2c]
   23abe: 01 00        	c.nop
   23ac0: 00 00 00 00  	v.nop
   23ac4: 00 00 00 00 00 00    	m.nop
   23aca: 31 25 c7 02  	l.st.w r7, [r4, 0x34]
   23ace: 01 00        	c.nop
   23ad0: 00 00 00 00  	v.nop
   23ad4: 00 00 00 00 00 00    	m.nop
   23ada: 31 21 c7 04  	l.st.w r7, [r4, 0x44]
   23ade: 01 00        	c.nop
   23ae0: 00 00 00 00  	v.nop
   23ae4: 00 00 00 00 00 00    	m.nop
   23aea: 31 23 c7 04  	l.st.w r7, [r4, 0x4c]
   23aee: 01 00        	c.nop
   23af0: 00 00 00 00  	v.nop
   23af4: 00 00 00 00 00 00    	m.nop
   23afa: 31 25 c7 04  	l.st.w r7, [r4, 0x54]
   23afe: 01 00        	c.nop
   23b00: 00 00 00 00  	v.nop
   23b04: 00 00 00 00 00 00    	m.nop
   23b0a: 31 27 c7 04  	l.st.w r7, [r4, 0x5c]
   23b0e: 01 00        	c.nop
   23b10: 00 00 00 00  	v.nop
   23b14: 00 00 00 00 00 00    	m.nop
   23b1a: 31 21 c7 06  	l.st.w r7, [r4, 0x64]
   23b1e: 01 00        	c.nop
   23b20: 00 00 00 00  	v.nop
   23b24: 00 00 00 00 00 00    	m.nop
   23b2a: 31 23 c7 06  	l.st.w r7, [r4, 0x6c]
   23b2e: 01 00        	c.nop
   23b30: 00 00 00 00  	v.nop
   23b34: 00 00 00 00 00 00    	m.nop
   23b3a: 31 25 c7 06  	l.st.w r7, [r4, 0x74]
   23b3e: 01 00        	c.nop
   23b40: 00 00 00 00  	v.nop
   23b44: 00 00 00 00 00 00    	m.nop
   23b4a: 6f 3e 7f 1f  	l.ldi16.u r7, 0xfff9
   23b4e: 01 00        	c.nop
   23b50: 00 00 00 00  	v.nop
   23b54: 00 00 00 00 00 00    	m.nop
   23b5a: 6f 3e ff 1e  	l.ldi_hi r7, 0xfff9
   23b5e: 01 00        	c.nop
   23b60: 00 00 00 00  	v.nop
   23b64: 00 00 00 00 00 00    	m.nop
   23b6a: 31 26 c7 02  	l.st.w r7, [r4, 0x38]
   23b6e: 01 00        	c.nop
   23b70: 00 00 00 00  	v.nop
   23b74: 00 00 00 00 00 00    	m.nop
   23b7a: ef 3f 6f 1f  	l.ldi16.u r7, 0xfdff
   23b7e: 01 00        	c.nop
   23b80: 00 00 00 00  	v.nop
   23b84: 00 00 00 00 00 00    	m.nop
   23b8a: ef 3f ef 1e  	l.ldi_hi r7, 0xfdff
   23b8e: 01 00        	c.nop
   23b90: 00 00 00 00  	v.nop
   23b94: 00 00 00 00 00 00    	m.nop
   23b9a: 31 26 c7 06  	l.st.w r7, [r4, 0x78]
   23b9e: 01 00        	c.nop
   23ba0: 00 00 00 00  	v.nop
   23ba4: 00 00 00 00 00 00    	m.nop
   23baa: 31 20 c8 00  	l.st.w r8, [r4]
   23bae: 01 00        	c.nop
   23bb0: 00 00 00 00  	v.nop
   23bb4: 00 00 00 00 00 00    	m.nop
   23bba: 31 22 c8 00  	l.st.w r8, [r4, 0x8]
   23bbe: 01 00        	c.nop
   23bc0: 00 00 00 00  	v.nop
   23bc4: 00 00 00 00 00 00    	m.nop
   23bca: 31 24 c8 00  	l.st.w r8, [r4, 0x10]
   23bce: 01 00        	c.nop
   23bd0: 00 00 00 00  	v.nop
   23bd4: 00 00 00 00 00 00    	m.nop
   23bda: 31 26 c8 00  	l.st.w r8, [r4, 0x18]
   23bde: 01 00        	c.nop
   23be0: 00 00 00 00  	v.nop
   23be4: 00 00 00 00 00 00    	m.nop
   23bea: 31 20 c8 02  	l.st.w r8, [r4, 0x20]
   23bee: 01 00        	c.nop
   23bf0: 00 00 00 00  	v.nop
   23bf4: 00 00 00 00 00 00    	m.nop
   23bfa: 31 22 c8 02  	l.st.w r8, [r4, 0x28]
   23bfe: 01 00        	c.nop
   23c00: 00 00 00 00  	v.nop
   23c04: 00 00 00 00 00 00    	m.nop
   23c0a: 31 24 c8 02  	l.st.w r8, [r4, 0x30]
   23c0e: 01 00        	c.nop
   23c10: 00 00 00 00  	v.nop
   23c14: 00 00 00 00 00 00    	m.nop
   23c1a: 31 20 c8 04  	l.st.w r8, [r4, 0x40]
   23c1e: 01 00        	c.nop
   23c20: 00 00 00 00  	v.nop
   23c24: 00 00 00 00 00 00    	m.nop
   23c2a: 31 22 c8 04  	l.st.w r8, [r4, 0x48]
   23c2e: 01 00        	c.nop
   23c30: 00 00 00 00  	v.nop
   23c34: 00 00 00 00 00 00    	m.nop
   23c3a: 31 24 c8 04  	l.st.w r8, [r4, 0x50]
   23c3e: 01 00        	c.nop
   23c40: 00 00 00 00  	v.nop
   23c44: 00 00 00 00 00 00    	m.nop
   23c4a: 31 26 c8 04  	l.st.w r8, [r4, 0x58]
   23c4e: 01 00        	c.nop
   23c50: 00 00 00 00  	v.nop
   23c54: 00 00 00 00 00 00    	m.nop
   23c5a: 31 20 c8 06  	l.st.w r8, [r4, 0x60]
   23c5e: 01 00        	c.nop
   23c60: 00 00 00 00  	v.nop
   23c64: 00 00 00 00 00 00    	m.nop
   23c6a: 31 22 c8 06  	l.st.w r8, [r4, 0x68]
   23c6e: 01 00        	c.nop
   23c70: 00 00 00 00  	v.nop
   23c74: 00 00 00 00 00 00    	m.nop
   23c7a: 31 24 c8 06  	l.st.w r8, [r4, 0x70]
   23c7e: 01 00        	c.nop
   23c80: 00 00 00 00  	v.nop
   23c84: 00 00 00 00 00 00    	m.nop
   23c8a: 31 27 c8 02  	l.st.w r8, [r4, 0x3c]
   23c8e: 01 00        	c.nop
   23c90: 00 00 00 00  	v.nop
   23c94: 00 00 00 00 00 00    	m.nop
   23c9a: 31 27 c8 06  	l.st.w r8, [r4, 0x7c]
   23c9e: 01 44        	<unknown>
   23ca0: 00 00 00 00  	v.nop
   23ca4: 00 00 00 00 00 00    	m.nop
   23caa: 2f 38 08 21  	l.ldi16.s r7, 0x100
   23cae: 01 00        	c.nop
   23cb0: 00 00 00 00  	v.nop
   23cb4: 00 00 00 00 00 00    	m.nop
   23cba: 31 21 cf 08  	l.st.w r15, [r4, 0x84]
   23cbe: 01 00        	c.nop
   23cc0: 00 00 00 00  	v.nop
   23cc4: 00 00 00 00 00 00    	m.nop
   23cca: 6f 58 00 00  	l.ldi_lo r11, 0x1
   23cce: 01 00        	c.nop
   23cd0: 00 00 00 00  	v.nop
   23cd4: 00 00 00 00 00 00    	m.nop
   23cda: 6f 58 80 00  	l.ldi_hi r11, 0x1
   23cde: 01 00        	c.nop
   23ce0: 00 00 00 00  	v.nop
   23ce4: 00 00 00 00 00 00    	m.nop
   23cea: e3 3d 86 03  	l.srli r23, r7, 0x6
   23cee: 01 00        	c.nop
   23cf0: 00 00 00 00  	v.nop
   23cf4: 00 00 00 00 00 00    	m.nop
   23cfa: 67 40 80 00  	l.slti.u r8, 0x1
   23cfe: 01 00        	c.nop
   23d00: 00 00 00 00  	v.nop
   23d04: 00 00 00 00 00 00    	m.nop
   23d0a: ef 60 00 00  	l.ldi_lo r12, 0x3
   23d0e: 01 00        	c.nop
   23d10: 00 00 00 00  	v.nop
   23d14: 00 00 00 00 00 00    	m.nop
   23d1a: ef 60 80 00  	l.ldi_hi r12, 0x3
   23d1e: 01 00        	c.nop
   23d20: 00 00 00 00  	v.nop
   23d24: 00 00 00 00 00 00    	m.nop
   23d2a: 25 63 cc 02  	l.xorix r12, r12, 0xc
   23d2e: 01 00        	c.nop
   23d30: 00 00 00 00  	v.nop
   23d34: 00 00 00 00 00 00    	m.nop
   23d3a: 00 00        	s.nop16
   23d3c: 00 00        	s.nop16
   23d3e: 4b 86        	<unknown>
   23d40: 00 00 00 00  	v.nop
   23d44: 00 00 00 00 00 00    	m.nop
   23d4a: 23 3b 85 03  	l.srli r12, r7, 0x5
   23d4e: 01 00        	c.nop
   23d50: 00 00 00 00  	v.nop
   23d54: 00 00 00 00 00 00    	m.nop
   23d5a: 63 63 10 03  	l.slli r13, r12, 0x10
   23d5e: 01 00        	c.nop
   23d60: 00 00 00 00  	v.nop
   23d64: 00 00 00 00 00 00    	m.nop
   23d6a: 25 63 0d 02  	l.orix r12, r12, 0xd
   23d6e: 01 00        	c.nop
   23d70: 00 00 00 00  	v.nop
   23d74: 00 00 00 00 00 00    	m.nop
   23d7a: 00 00        	s.nop16
   23d7c: 00 00        	s.nop16
   23d7e: 0f 56        	<unknown>
   23d80: 00 00 00 00  	v.nop
   23d84: 00 00 00 00 00 00    	m.nop
   23d8a: 00 00        	s.nop16
   23d8c: 00 00        	s.nop16
   23d8e: 2b 56        	<unknown>
   23d90: 00 00 00 00  	v.nop
   23d94: 00 00 00 00 00 00    	m.nop
   23d9a: 23 bb 10 03  	l.slli r12, r23, 0x10
   23d9e: 01 00        	c.nop
   23da0: 00 00 00 00  	v.nop
   23da4: 00 00 00 00 00 00    	m.nop
   23daa: 25 63 17 02  	l.orix r12, r12, 0x17
   23dae: 01 00        	c.nop
   23db0: 00 00 00 00  	v.nop
   23db4: 00 00 00 00 00 00    	m.nop
   23dba: 00 00        	s.nop16
   23dbc: 00 00        	s.nop16
   23dbe: 73 56        	<unknown>
   23dc0: 00 00 00 00  	v.nop
   23dc4: 00 00 00 00 00 00    	m.nop
   23dca: ef 67 01 01  	l.ldi16.u r12, 0x3f
   23dce: 01 00        	c.nop
   23dd0: 00 00 00 00  	v.nop
   23dd4: 00 00 00 00 00 00    	m.nop
   23dda: 6f 68 00 01  	l.ldi16.u r13, 0x1
   23dde: 01 00        	c.nop
   23de0: 00 00 00 00  	v.nop
   23de4: 00 00 00 00 00 00    	m.nop
   23dea: 23 63 17 19  	<unknown>
   23dee: 01 00        	c.nop
   23df0: 00 00 00 00  	v.nop
   23df4: 00 00 00 00 00 00    	m.nop
   23dfa: 23 6b 8c 18  	<unknown>
   23dfe: 01 00        	c.nop
   23e00: 00 00 00 00  	v.nop
   23e04: 00 00 00 00 00 00    	m.nop
   23e0a: 63 63 10 03  	l.slli r13, r12, 0x10
   23e0e: 01 00        	c.nop
   23e10: 00 00 00 00  	v.nop
   23e14: 00 00 00 00 00 00    	m.nop
   23e1a: 23 6b 90 03  	l.srli r12, r13, 0x10
   23e1e: 01 00        	c.nop
   23e20: 00 00 00 00  	v.nop
   23e24: 00 00 00 00 00 00    	m.nop
   23e2a: 25 63 0d 02  	l.orix r12, r12, 0xd
   23e2e: 01 00        	c.nop
   23e30: 00 00 00 00  	v.nop
   23e34: 00 00 00 00 00 00    	m.nop
   23e3a: 00 00        	s.nop16
   23e3c: 00 00        	s.nop16
   23e3e: 77 56        	<unknown>
   23e40: 00 00 00 00  	v.nop
   23e44: 00 00 00 00 00 00    	m.nop
   23e4a: 6f 60 00 00  	l.ldi_lo r12, 0x1
   23e4e: 01 00        	c.nop
   23e50: 00 00 00 00  	v.nop
   23e54: 00 00 00 00 00 00    	m.nop
   23e5a: 6f 60 80 00  	l.ldi_hi r12, 0x1
   23e5e: 01 00        	c.nop
   23e60: 00 00 00 00  	v.nop
   23e64: 00 00 00 00 00 00    	m.nop
   23e6a: 00 00        	s.nop16
   23e6c: 00 00        	s.nop16
   23e6e: 47 56        	<unknown>
   23e70: 00 00 00 00  	v.nop
   23e74: 00 00 00 00 00 00    	m.nop
   23e7a: 33 43 00 00  	l.uext.b r12, r8
   23e7e: 01 00        	c.nop
   23e80: 00 00 00 00  	v.nop
   23e84: 00 00 00 00 00 00    	m.nop
   23e8a: 00 00        	s.nop16
   23e8c: 00 00        	s.nop16
   23e8e: 03 8e        	<unknown>
   23e90: 00 00 00 00  	v.nop
   23e94: 00 00 00 00 00 00    	m.nop
   23e9a: 00 00        	s.nop16
   23e9c: 00 00        	s.nop16
   23e9e: 23 86        	<unknown>
   23ea0: 00 00 00 00  	v.nop
   23ea4: 00 00 00 00 00 00    	m.nop
   23eaa: 00 00        	s.nop16
   23eac: 00 00        	s.nop16
   23eae: 17 86        	<unknown>
   23eb0: 00 00 00 00  	v.nop
   23eb4: 00 00 00 00 00 00    	m.nop
   23eba: 00 00        	s.nop16
   23ebc: 00 00        	s.nop16
   23ebe: 1b 86        	<unknown>
   23ec0: 00 00 00 00  	v.nop
   23ec4: 00 00 00 00 00 00    	m.nop
   23eca: 33 45 00 00  	l.uext.b r20, r8
   23ece: 1f 86        	<unknown>
   23ed0: 00 00 00 00  	v.nop
   23ed4: 00 00 00 00 00 00    	m.nop
   23eda: a3 31 86 0b  	l.srai r6, r6, 0x6
   23ede: 01 00        	c.nop
   23ee0: 00 00 00 00  	v.nop
   23ee4: 00 00 00 00 00 00    	m.nop
   23eea: 2f 60 00 01  	l.ldi16.u r12, 0x0
   23eee: 01 00        	c.nop
   23ef0: 00 00 00 00  	v.nop
   23ef4: 00 00 00 00 00 00    	m.nop
   23efa: 23 63 06 10  	<unknown>
   23efe: 01 00        	c.nop
   23f00: 00 00 00 00  	v.nop
   23f04: 00 00 00 00 00 00    	m.nop
   23f0a: 23 63 10 03  	l.slli r12, r12, 0x10
   23f0e: 01 00        	c.nop
   23f10: 00 00 00 00  	v.nop
   23f14: 00 00 00 00 00 00    	m.nop
   23f1a: 25 63 06 02  	l.orix r12, r12, 0x6
   23f1e: 01 00        	c.nop
   23f20: 00 00 00 00  	v.nop
   23f24: 00 00 00 00 00 00    	m.nop
   23f2a: 00 00        	s.nop16
   23f2c: 00 00        	s.nop16
   23f2e: 03 56        	<unknown>
   23f30: 00 00 00 00  	v.nop
   23f34: 00 00 00 00 00 00    	m.nop
   23f3a: 63 29 86 0b  	l.srai r5, r5, 0x6
   23f3e: 01 00        	c.nop
   23f40: 00 00 00 00  	v.nop
   23f44: 00 00 00 00 00 00    	m.nop
   23f4a: 63 29 0b 11  	<unknown>
   23f4e: 01 00        	c.nop
   23f50: 00 00 00 00  	v.nop
   23f54: 00 00 00 00 00 00    	m.nop
   23f5a: 23 bb 07 11  	<unknown>
   23f5e: 01 00        	c.nop
   23f60: 00 00 00 00  	v.nop
   23f64: 00 00 00 00 00 00    	m.nop
   23f6a: 23 63 10 03  	l.slli r12, r12, 0x10
   23f6e: 01 00        	c.nop
   23f70: 00 00 00 00  	v.nop
   23f74: 00 00 00 00 00 00    	m.nop
   23f7a: 23 63 05 10  	<unknown>
   23f7e: 01 00        	c.nop
   23f80: 00 00 00 00  	v.nop
   23f84: 00 00 00 00 00 00    	m.nop
   23f8a: 00 00        	s.nop16
   23f8c: 00 00        	s.nop16
   23f8e: 23 56        	<unknown>
   23f90: 00 00 00 00  	v.nop
   23f94: 00 00 00 00 00 00    	m.nop
   23f9a: 23 bb 81 03  	l.srli r12, r23, 0x1
   23f9e: 01 00        	c.nop
   23fa0: 00 00 00 00  	v.nop
   23fa4: 00 00 00 00 00 00    	m.nop
   23faa: 23 63 10 03  	l.slli r12, r12, 0x10
   23fae: 01 00        	c.nop
   23fb0: 00 00 00 00  	v.nop
   23fb4: 00 00 00 00 00 00    	m.nop
   23fba: 00 00        	s.nop16
   23fbc: 00 00        	s.nop16
   23fbe: 63 56        	<unknown>
   23fc0: 00 00 00 00  	v.nop
   23fc4: 00 00 00 00 00 00    	m.nop
   23fca: 23 bb 81 03  	l.srli r12, r23, 0x1
   23fce: 01 00        	c.nop
   23fd0: 00 00 00 00  	v.nop
   23fd4: 00 00 00 00 00 00    	m.nop
   23fda: 23 63 10 03  	l.slli r12, r12, 0x10
   23fde: 01 00        	c.nop
   23fe0: 00 00 00 00  	v.nop
   23fe4: 00 00 00 00 00 00    	m.nop
   23fea: 00 00        	s.nop16
   23fec: 00 00        	s.nop16
   23fee: 43 56        	<unknown>
   23ff0: 00 00 00 00  	v.nop
   23ff4: 00 00 00 00 00 00    	m.nop
   23ffa: f3 23 00 00  	l.uext.b r15, r4
   23ffe: 01 00        	c.nop
   24000: 00 00 00 00  	v.nop
   24004: 00 00 00 00 00 00    	m.nop
   2400a: 00 00        	s.nop16
   2400c: 00 00        	s.nop16
   2400e: 98 43        	<unknown>
   24010: 00 00 00 00  	v.nop
   24014: 00 00 00 00 00 00    	m.nop
   2401a: 00 00        	s.nop16
   2401c: 00 00        	s.nop16
   2401e: d4 43        	<unknown>
   24020: e8 51 41 01  	<unknown>
   24024: 00 00 00 00 00 00    	m.nop
   2402a: 00 00        	s.nop16
   2402c: 00 00        	s.nop16
   2402e: 2f 57        	<unknown>
   24030: e8 51 41 09  	<unknown>
   24034: 00 00 00 00 00 00    	m.nop
   2403a: 00 00        	s.nop16
   2403c: 00 00        	s.nop16
   2403e: 98 47        	<unknown>
   24040: e8 51 41 11  	<unknown>
   24044: 00 00 00 00 00 00    	m.nop
   2404a: 00 00        	s.nop16
   2404c: 00 00        	s.nop16
   2404e: 93 56        	<unknown>
   24050: e8 51 41 19  	<unknown>
   24054: 00 00 00 00 00 00    	m.nop
   2405a: 00 00        	s.nop16
   2405c: 00 00        	s.nop16
   2405e: d4 47        	<unknown>
   24060: e8 51 41 21  	<unknown>
   24064: 00 00 00 00 00 00    	m.nop
   2406a: 00 00        	s.nop16
   2406c: 00 00        	s.nop16
   2406e: 01 00        	c.nop
   24070: e8 51 41 29  	<unknown>
   24074: 00 00 00 00 00 00    	m.nop
   2407a: 00 00        	s.nop16
   2407c: 00 00        	s.nop16
   2407e: 01 00        	c.nop
   24080: e8 51 41 31  	<unknown>
   24084: 00 00 00 00 00 00    	m.nop
   2408a: 00 00        	s.nop16
   2408c: 00 00        	s.nop16
   2408e: 01 00        	c.nop
   24090: e8 51 41 39  	<unknown>
   24094: 00 00 00 00 00 00    	m.nop
   2409a: 00 00        	s.nop16
   2409c: 00 00        	s.nop16
   2409e: 01 00        	c.nop
   240a0: e8 51 41 41  	<unknown>
   240a4: 00 00 00 00 00 00    	m.nop
   240aa: 00 00        	s.nop16
   240ac: 00 00        	s.nop16
   240ae: 01 00        	c.nop
   240b0: e8 51 41 49  	<unknown>
   240b4: 00 00 00 00 00 00    	m.nop
   240ba: 00 00        	s.nop16
   240bc: 00 00        	s.nop16
   240be: 01 00        	c.nop
   240c0: e8 51 41 51  	<unknown>
   240c4: 00 00 00 00 00 00    	m.nop
   240ca: 00 00        	s.nop16
   240cc: 00 00        	s.nop16
   240ce: 01 00        	c.nop
   240d0: e8 51 41 59  	<unknown>
   240d4: 00 00 00 00 00 00    	m.nop
   240da: 00 00        	s.nop16
   240dc: 00 00        	s.nop16
   240de: 01 00        	c.nop
   240e0: e8 51 41 61  	<unknown>
   240e4: 00 00 00 00 00 00    	m.nop
   240ea: 00 00        	s.nop16
   240ec: 00 00        	s.nop16
   240ee: 01 00        	c.nop
   240f0: e8 51 41 69  	<unknown>
   240f4: 00 00 00 00 00 00    	m.nop
   240fa: 00 00        	s.nop16
   240fc: 00 00        	s.nop16
   240fe: 01 00        	c.nop
   24100: e8 51 41 71  	<unknown>
   24104: 00 00 00 00 00 00    	m.nop
   2410a: 00 00        	s.nop16
   2410c: 00 00        	s.nop16
   2410e: 01 00        	c.nop
   24110: e8 51 41 79  	<unknown>
   24114: 00 00 00 00 00 00    	m.nop
   2411a: 00 00        	s.nop16
   2411c: 00 00        	s.nop16
   2411e: 01 00        	c.nop
   24120: e8 51 41 81  	<unknown>
   24124: 00 00 00 00 00 00    	m.nop
   2412a: 00 00        	s.nop16
   2412c: 00 00        	s.nop16
   2412e: 01 00        	c.nop
   24130: e8 51 41 89  	<unknown>
   24134: 00 00 00 00 00 00    	m.nop
   2413a: 00 00        	s.nop16
   2413c: 00 00        	s.nop16
   2413e: 01 00        	c.nop
   24140: e8 51 41 91  	<unknown>
   24144: 00 00 00 00 00 00    	m.nop
   2414a: 00 00        	s.nop16
   2414c: 00 00        	s.nop16
   2414e: 01 00        	c.nop
   24150: e8 51 41 99  	<unknown>
   24154: 00 00 00 00 00 00    	m.nop
   2415a: 00 00        	s.nop16
   2415c: 00 00        	s.nop16
   2415e: 01 00        	c.nop
   24160: e8 51 41 a1  	<unknown>
   24164: 00 00 00 00 00 00    	m.nop
   2416a: 00 00        	s.nop16
   2416c: 00 00        	s.nop16
   2416e: 01 00        	c.nop
   24170: e8 51 41 a9  	<unknown>
   24174: 00 00 00 00 00 00    	m.nop
   2417a: 00 00        	s.nop16
   2417c: 00 00        	s.nop16
   2417e: 01 00        	c.nop
   24180: e8 51 41 b1  	<unknown>
   24184: 00 00 00 00 00 00    	m.nop
   2418a: 00 00        	s.nop16
   2418c: 00 00        	s.nop16
   2418e: 01 00        	c.nop
   24190: e8 51 41 b9  	<unknown>
   24194: 00 00 00 00 00 00    	m.nop
   2419a: 00 00        	s.nop16
   2419c: 00 00        	s.nop16
   2419e: 01 00        	c.nop
   241a0: e8 51 41 c1  	<unknown>
   241a4: 00 00 00 00 00 00    	m.nop
   241aa: 00 00        	s.nop16
   241ac: 00 00        	s.nop16
   241ae: 01 00        	c.nop
   241b0: e8 51 41 c9  	<unknown>
   241b4: 00 00 00 00 00 00    	m.nop
   241ba: 00 00        	s.nop16
   241bc: 00 00        	s.nop16
   241be: 01 00        	c.nop
   241c0: e8 51 41 d1  	<unknown>
   241c4: 00 00 00 00 00 00    	m.nop
   241ca: 00 00        	s.nop16
   241cc: 00 00        	s.nop16
   241ce: 01 00        	c.nop
   241d0: e8 51 41 d9  	<unknown>
   241d4: 00 00 00 00 00 00    	m.nop
   241da: 00 00        	s.nop16
   241dc: 00 00        	s.nop16
   241de: 01 00        	c.nop
   241e0: e8 51 41 e1  	<unknown>
   241e4: 00 00 00 00 00 00    	m.nop
   241ea: 00 00        	s.nop16
   241ec: 00 00        	s.nop16
   241ee: 01 00        	c.nop
   241f0: e8 51 41 e9  	<unknown>
   241f4: 00 00 00 00 00 00    	m.nop
   241fa: 00 00        	s.nop16
   241fc: 00 00        	s.nop16
   241fe: 01 00        	c.nop
   24200: e8 51 41 f1  	<unknown>
   24204: 00 00 00 00 00 00    	m.nop
   2420a: 00 00        	s.nop16
   2420c: 00 00        	s.nop16
   2420e: 01 00        	c.nop
   24210: e8 51 41 f9  	<unknown>
   24214: 00 00 00 00 00 00    	m.nop
   2421a: 00 00        	s.nop16
   2421c: 00 00        	s.nop16
   2421e: 01 00        	c.nop
   24220: 00 00 00 00  	v.nop
   24224: 00 00 00 00 00 00    	m.nop
   2422a: 1e 58        	s.tar.add ta11, ta0
   2422c: 00 00        	s.nop16
   2422e: 01 00        	c.nop
   24230: 00 00 00 00  	v.nop
   24234: 00 00 00 00 00 00    	m.nop
   2423a: 28 18        	<unknown>
   2423c: 00 00        	s.nop16
   2423e: 01 00        	c.nop
   24240: 00 00 00 00  	v.nop
   24244: 00 00 00 00 00 00    	m.nop
   2424a: 68 18        	<unknown>
   2424c: 00 00        	s.nop16
   2424e: 01 00        	c.nop
   24250: 00 00 00 00  	v.nop
   24254: 00 00 00 00 00 00    	m.nop
   2425a: a8 18        	<unknown>
   2425c: 00 00        	s.nop16
   2425e: 01 00        	c.nop
   24260: 00 00 00 00  	v.nop
   24264: 00 00 00 00 00 00    	m.nop
   2426a: e8 18        	<unknown>
   2426c: 00 00        	s.nop16
   2426e: 01 00        	c.nop
   24270: 00 00 00 00  	v.nop
   24274: 00 00 00 00 00 00    	m.nop
   2427a: 28 19        	<unknown>
   2427c: 00 00        	s.nop16
   2427e: 01 00        	c.nop
   24280: 00 00 00 00  	v.nop
   24284: 00 00 00 00 00 00    	m.nop
   2428a: 68 19        	<unknown>
   2428c: 00 00        	s.nop16
   2428e: 01 00        	c.nop
   24290: 00 00 00 00  	v.nop
   24294: 00 00 00 00 00 00    	m.nop
   2429a: a8 19        	<unknown>
   2429c: 00 00        	s.nop16
   2429e: 01 00        	c.nop
   242a0: 00 00 00 00  	v.nop
   242a4: 00 00 00 00 00 00    	m.nop
   242aa: e8 19        	<unknown>
   242ac: 00 00        	s.nop16
   242ae: 01 00        	c.nop
   242b0: 00 00 00 00  	v.nop
   242b4: 00 00 00 00 00 00    	m.nop
   242ba: 28 1a        	<unknown>
   242bc: 00 00        	s.nop16
   242be: 01 00        	c.nop
   242c0: 00 00 00 00  	v.nop
   242c4: 00 00 00 00 00 00    	m.nop
   242ca: 68 1a        	<unknown>
   242cc: 00 00        	s.nop16
   242ce: 01 00        	c.nop
   242d0: 00 00 00 00  	v.nop
   242d4: 00 00 00 00 00 00    	m.nop
   242da: a8 1a        	<unknown>
   242dc: 00 00        	s.nop16
   242de: 01 00        	c.nop
   242e0: 00 00 00 00  	v.nop
   242e4: 00 00 00 00 00 00    	m.nop
   242ea: e8 1a        	<unknown>
   242ec: 00 00        	s.nop16
   242ee: 01 00        	c.nop
   242f0: 00 00 00 00  	v.nop
   242f4: 00 00 00 00 00 00    	m.nop
   242fa: 28 1b        	<unknown>
   242fc: 00 00        	s.nop16
   242fe: 01 00        	c.nop
   24300: 00 00 00 00  	v.nop
   24304: 00 00 00 00 00 00    	m.nop
   2430a: 68 1b        	<unknown>
   2430c: 00 00        	s.nop16
   2430e: 01 00        	c.nop
   24310: 00 00 00 00  	v.nop
   24314: 00 00 00 00 00 00    	m.nop
   2431a: a8 1b        	<unknown>
   2431c: 00 00        	s.nop16
   2431e: 01 00        	c.nop
   24320: 00 00 00 00  	v.nop
   24324: 00 00 00 00 00 00    	m.nop
   2432a: e8 1b        	<unknown>
   2432c: 00 00        	s.nop16
   2432e: 01 00        	c.nop
   24330: 00 00 00 00  	v.nop
   24334: 00 00 00 00 00 00    	m.nop
   2433a: 23 3b 85 03  	l.srli r12, r7, 0x5
   2433e: 01 00        	c.nop
   24340: 00 00 00 00  	v.nop
   24344: 00 00 00 00 00 00    	m.nop
   2434a: 21 63 01 30  	l.subi.s r12, r12, 0x1
   2434e: 01 00        	c.nop
   24350: 00 00 00 00  	v.nop
   24354: 00 00 00 00 00 00    	m.nop
   2435a: 00 00        	s.nop16
   2435c: 00 00        	s.nop16
   2435e: 6b 82        	<unknown>
   24360: 00 00 00 00  	v.nop
   24364: 00 00 00 00 00 00    	m.nop
   2436a: 3f 08 82 00  	<unknown>
   2436e: 01 00        	c.nop
   24370: e8 51 41 03  	<unknown>
   24374: 24 00 20 00 c4 02    	m.nop
   2437a: 28 18        	<unknown>
   2437c: 00 00        	s.nop16
   2437e: 2f 57        	<unknown>
   24380: e8 51 41 0b  	<unknown>
   24384: 24 01 21 00 c4 02    	m.nop
   2438a: 68 18        	<unknown>
   2438c: 00 00        	s.nop16
   2438e: 98 4b        	<unknown>
   24390: e8 51 41 13  	<unknown>
   24394: 24 02 22 00 c4 02    	m.nop
   2439a: a8 18        	<unknown>
   2439c: 00 00        	s.nop16
   2439e: 01 00        	c.nop
   243a0: e8 51 41 1b  	<unknown>
   243a4: 24 03 23 00 c4 02    	m.nop
   243aa: e8 18        	<unknown>
   243ac: 00 00        	s.nop16
   243ae: 01 00        	c.nop
   243b0: e8 51 41 23  	<unknown>
   243b4: 24 04 24 00 c4 02    	m.nop
   243ba: 28 19        	<unknown>
   243bc: 00 00        	s.nop16
   243be: 01 00        	c.nop
   243c0: e8 51 41 2b  	<unknown>
   243c4: 24 05 25 00 c4 02    	m.nop
   243ca: 68 19        	<unknown>
   243cc: 00 00        	s.nop16
   243ce: 01 00        	c.nop
   243d0: e8 51 41 33  	<unknown>
   243d4: 24 06 26 00 c4 02    	m.nop
   243da: a8 19        	<unknown>
   243dc: 00 00        	s.nop16
   243de: 01 00        	c.nop
   243e0: e8 51 41 3b  	<unknown>
   243e4: 24 07 27 00 c4 02    	m.nop
   243ea: e8 19        	<unknown>
   243ec: 00 00        	s.nop16
   243ee: 01 00        	c.nop
   243f0: e8 51 41 43  	<unknown>
   243f4: 24 08 28 00 c4 02    	m.nop
   243fa: 28 1a        	<unknown>
   243fc: 00 00        	s.nop16
   243fe: 01 00        	c.nop
   24400: e8 51 41 4b  	<unknown>
   24404: 24 09 29 00 c4 02    	m.nop
   2440a: 68 1a        	<unknown>
   2440c: 00 00        	s.nop16
   2440e: 01 00        	c.nop
   24410: e8 51 41 53  	<unknown>
   24414: 24 0a 2a 00 c4 02    	m.nop
   2441a: a8 1a        	<unknown>
   2441c: 00 00        	s.nop16
   2441e: 01 00        	c.nop
   24420: e8 51 41 5b  	<unknown>
   24424: 24 0b 2b 00 c4 02    	m.nop
   2442a: e8 1a        	<unknown>
   2442c: 00 00        	s.nop16
   2442e: 01 00        	c.nop
   24430: e8 51 41 63  	<unknown>
   24434: 24 0c 2c 00 c4 02    	m.nop
   2443a: 28 1b        	<unknown>
   2443c: 00 00        	s.nop16
   2443e: 01 00        	c.nop
   24440: e8 51 41 6b  	<unknown>
   24444: 24 0d 2d 00 c4 02    	m.nop
   2444a: 68 1b        	<unknown>
   2444c: 00 00        	s.nop16
   2444e: 01 00        	c.nop
   24450: e8 51 41 73  	<unknown>
   24454: 24 0e 2e 00 c4 02    	m.nop
   2445a: a8 1b        	<unknown>
   2445c: 00 00        	s.nop16
   2445e: 01 00        	c.nop
   24460: e8 51 41 7b  	<unknown>
   24464: 24 0f 2f 00 c4 02    	m.nop
   2446a: e8 1b        	<unknown>
   2446c: 00 00        	s.nop16
   2446e: 01 00        	c.nop
   24470: e8 51 41 83  	<unknown>
   24474: 24 10 20 00 c4 02    	m.nop
   2447a: 28 18        	<unknown>
   2447c: 00 00        	s.nop16
   2447e: 01 00        	c.nop
   24480: e8 51 41 8b  	<unknown>
   24484: 24 11 21 00 c4 02    	m.nop
   2448a: 68 18        	<unknown>
   2448c: 00 00        	s.nop16
   2448e: 01 00        	c.nop
   24490: e8 51 41 93  	<unknown>
   24494: 24 12 22 00 c4 02    	m.nop
   2449a: a8 18        	<unknown>
   2449c: 00 00        	s.nop16
   2449e: 01 00        	c.nop
   244a0: e8 51 41 9b  	<unknown>
   244a4: 24 13 23 00 c4 02    	m.nop
   244aa: e8 18        	<unknown>
   244ac: 00 00        	s.nop16
   244ae: 01 00        	c.nop
   244b0: e8 51 41 a3  	<unknown>
   244b4: 24 14 24 00 c4 02    	m.nop
   244ba: 28 19        	<unknown>
   244bc: 00 00        	s.nop16
   244be: 01 00        	c.nop
   244c0: e8 51 41 ab  	<unknown>
   244c4: 24 15 25 00 c4 02    	m.nop
   244ca: 68 19        	<unknown>
   244cc: 00 00        	s.nop16
   244ce: 01 00        	c.nop
   244d0: e8 51 41 b3  	<unknown>
   244d4: 24 16 26 00 c4 02    	m.nop
   244da: a8 19        	<unknown>
   244dc: 00 00        	s.nop16
   244de: 01 00        	c.nop
   244e0: e8 51 41 bb  	<unknown>
   244e4: 24 17 27 00 c4 02    	m.nop
   244ea: e8 19        	<unknown>
   244ec: 00 00        	s.nop16
   244ee: 01 00        	c.nop
   244f0: e8 51 41 c3  	<unknown>
   244f4: 24 18 28 00 c4 02    	m.nop
   244fa: 28 1a        	<unknown>
   244fc: 00 00        	s.nop16
   244fe: 01 00        	c.nop
   24500: e8 51 41 cb  	<unknown>
   24504: 24 19 29 00 c4 02    	m.nop
   2450a: 68 1a        	<unknown>
   2450c: 00 00        	s.nop16
   2450e: 01 00        	c.nop
   24510: e8 51 41 d3  	<unknown>
   24514: 24 1a 2a 00 c4 02    	m.nop
   2451a: a8 1a        	<unknown>
   2451c: 00 00        	s.nop16
   2451e: 01 00        	c.nop
   24520: e8 51 41 db  	<unknown>
   24524: 24 1b 2b 00 c4 02    	m.nop
   2452a: e8 1a        	<unknown>
   2452c: 00 00        	s.nop16
   2452e: 01 00        	c.nop
   24530: e8 51 41 e3  	<unknown>
   24534: 24 1c 2c 00 c4 02    	m.nop
   2453a: 28 1b        	<unknown>
   2453c: 00 00        	s.nop16
   2453e: 01 00        	c.nop
   24540: e8 51 41 eb  	<unknown>
   24544: 24 1d 2d 00 c4 02    	m.nop
   2454a: 68 1b        	<unknown>
   2454c: 00 00        	s.nop16
   2454e: 01 00        	c.nop
   24550: e8 51 41 f3  	<unknown>
   24554: 24 1e 2e 00 c4 02    	m.nop
   2455a: a8 1b        	<unknown>
   2455c: 00 00        	s.nop16
   2455e: 01 00        	c.nop
   24560: e8 51 41 fb  	<unknown>
   24564: 24 1f 2f 00 c4 02    	m.nop
   2456a: e8 1b        	<unknown>
   2456c: 00 00        	s.nop16
   2456e: 01 00        	c.nop
   24570: 00 00 00 00  	v.nop
   24574: 00 00 00 00 00 00    	m.nop
   2457a: 1e 58        	s.tar.add ta11, ta0
   2457c: 00 00        	s.nop16
   2457e: 01 00        	c.nop
   24580: 00 00 00 00  	v.nop
   24584: 24 20 20 00 c4 02    	m.nop
   2458a: 28 18        	<unknown>
   2458c: 00 00        	s.nop16
   2458e: 01 00        	c.nop
   24590: 00 00 00 00  	v.nop
   24594: 24 21 21 00 c4 02    	m.nop
   2459a: 68 18        	<unknown>
   2459c: 00 00        	s.nop16
   2459e: 01 00        	c.nop
   245a0: 00 00 00 00  	v.nop
   245a4: 24 22 22 00 c4 02    	m.nop
   245aa: a8 18        	<unknown>
   245ac: 00 00        	s.nop16
   245ae: 01 00        	c.nop
   245b0: 00 00 00 00  	v.nop
   245b4: 24 23 23 00 c4 02    	m.nop
   245ba: e8 18        	<unknown>
   245bc: 00 00        	s.nop16
   245be: 01 00        	c.nop
   245c0: 00 00 00 00  	v.nop
   245c4: 24 24 24 00 c4 02    	m.nop
   245ca: 28 19        	<unknown>
   245cc: 00 00        	s.nop16
   245ce: 01 00        	c.nop
   245d0: 00 00 00 00  	v.nop
   245d4: 24 25 25 00 c4 02    	m.nop
   245da: 68 19        	<unknown>
   245dc: 00 00        	s.nop16
   245de: 01 00        	c.nop
   245e0: 00 00 00 00  	v.nop
   245e4: 24 26 26 00 c4 02    	m.nop
   245ea: a8 19        	<unknown>
   245ec: 00 00        	s.nop16
   245ee: 01 00        	c.nop
   245f0: 00 00 00 00  	v.nop
   245f4: 24 27 27 00 c4 02    	m.nop
   245fa: e8 19        	<unknown>
   245fc: 00 00        	s.nop16
   245fe: 01 00        	c.nop
   24600: 00 00 00 00  	v.nop
   24604: 24 28 28 00 c4 02    	m.nop
   2460a: 28 1a        	<unknown>
   2460c: 00 00        	s.nop16
   2460e: 01 00        	c.nop
   24610: 00 00 00 00  	v.nop
   24614: 24 29 29 00 c4 02    	m.nop
   2461a: 68 1a        	<unknown>
   2461c: 00 00        	s.nop16
   2461e: 01 00        	c.nop
   24620: 00 00 00 00  	v.nop
   24624: 24 2a 2a 00 c4 02    	m.nop
   2462a: a8 1a        	<unknown>
   2462c: 00 00        	s.nop16
   2462e: 01 00        	c.nop
   24630: 00 00 00 00  	v.nop
   24634: 24 2b 2b 00 c4 02    	m.nop
   2463a: e8 1a        	<unknown>
   2463c: 00 00        	s.nop16
   2463e: 01 00        	c.nop
   24640: 00 00 00 00  	v.nop
   24644: 24 2c 2c 00 c4 02    	m.nop
   2464a: 28 1b        	<unknown>
   2464c: 00 00        	s.nop16
   2464e: 01 00        	c.nop
   24650: 00 00 00 00  	v.nop
   24654: 24 2d 2d 00 c4 02    	m.nop
   2465a: 68 1b        	<unknown>
   2465c: 00 00        	s.nop16
   2465e: 01 00        	c.nop
   24660: 00 00 00 00  	v.nop
   24664: 24 2e 2e 00 c4 02    	m.nop
   2466a: a8 1b        	<unknown>
   2466c: 00 00        	s.nop16
   2466e: 01 00        	c.nop
   24670: 00 00 00 00  	v.nop
   24674: 24 2f 2f 00 c4 02    	m.nop
   2467a: e8 23        	<unknown>
   2467c: 00 00        	s.nop16
   2467e: 01 00        	c.nop
   24680: 00 00 00 00  	v.nop
   24684: 24 30 20 00 c4 02    	m.nop
   2468a: 28 18        	<unknown>
   2468c: 00 00        	s.nop16
   2468e: 93 56        	<unknown>
   24690: 00 00 00 00  	v.nop
   24694: 24 31 21 00 c4 02    	m.nop
   2469a: 68 18        	<unknown>
   2469c: 00 00        	s.nop16
   2469e: d4 4b        	<unknown>
   246a0: 00 00 00 00  	v.nop
   246a4: 24 32 22 00 c4 02    	m.nop
   246aa: a8 18        	<unknown>
   246ac: 00 00        	s.nop16
   246ae: a1 07        	<unknown>
   246b0: 00 00 00 00  	v.nop
   246b4: 24 33 23 00 c4 02    	m.nop
   246ba: e8 18        	<unknown>
   246bc: 00 00        	s.nop16
   246be: 01 00        	c.nop
   246c0: 00 00 00 00  	v.nop
   246c4: 24 34 24 00 c4 02    	m.nop
   246ca: 28 19        	<unknown>
   246cc: 00 00        	s.nop16
   246ce: 01 00        	c.nop
   246d0: 00 00 00 00  	v.nop
   246d4: 24 35 25 00 c4 02    	m.nop
   246da: 68 19        	<unknown>
   246dc: 00 00        	s.nop16
   246de: 01 00        	c.nop
   246e0: 00 00 00 00  	v.nop
   246e4: 24 36 26 00 c4 02    	m.nop
   246ea: a8 19        	<unknown>
   246ec: 00 00        	s.nop16
   246ee: 01 00        	c.nop
   246f0: 00 00 00 00  	v.nop
   246f4: 24 37 27 00 c4 02    	m.nop
   246fa: e8 19        	<unknown>
   246fc: 00 00        	s.nop16
   246fe: 01 00        	c.nop
   24700: 00 00 00 00  	v.nop
   24704: 24 38 28 00 c4 02    	m.nop
   2470a: 28 1a        	<unknown>
   2470c: 00 00        	s.nop16
   2470e: 01 00        	c.nop
   24710: 00 00 00 00  	v.nop
   24714: 24 39 29 00 c4 02    	m.nop
   2471a: 68 1a        	<unknown>
   2471c: 00 00        	s.nop16
   2471e: 01 00        	c.nop
   24720: 00 00 00 00  	v.nop
   24724: 24 3a 2a 00 c4 02    	m.nop
   2472a: a8 1a        	<unknown>
   2472c: 00 00        	s.nop16
   2472e: 01 00        	c.nop
   24730: 00 00 00 00  	v.nop
   24734: 24 3b 2b 00 c4 02    	m.nop
   2473a: e8 1a        	<unknown>
   2473c: 00 00        	s.nop16
   2473e: 01 00        	c.nop
   24740: 00 00 00 00  	v.nop
   24744: 24 3c 2c 00 c4 02    	m.nop
   2474a: 28 1b        	<unknown>
   2474c: 00 00        	s.nop16
   2474e: 01 00        	c.nop
   24750: 00 00 00 00  	v.nop
   24754: 24 3d 2d 00 c4 02    	m.nop
   2475a: 68 1b        	<unknown>
   2475c: 00 00        	s.nop16
   2475e: 01 00        	c.nop
   24760: 00 00 00 00  	v.nop
   24764: 24 3e 2e 00 c4 02    	m.nop
   2476a: a8 1b        	<unknown>
   2476c: 00 00        	s.nop16
   2476e: 01 00        	c.nop
   24770: 00 00 00 00  	v.nop
   24774: 24 3f 2f 00 c4 02    	m.nop
   2477a: e8 1b        	<unknown>
   2477c: 00 00        	s.nop16
   2477e: 01 46        	<unknown>
   24780: 40 00 80 00  	<unknown>
   24784: 00 00 00 00 00 00    	m.nop
   2478a: 00 00        	s.nop16
   2478c: 00 00        	s.nop16
   2478e: 4b 86        	<unknown>
   24790: 00 00 00 00  	v.nop
   24794: 00 00 00 00 00 00    	m.nop
   2479a: 21 a3 00 01  	l.addi.u r12, r20, 0x100
   2479e: f9 1b        	<unknown>
   247a0: 00 00 00 00  	v.nop
   247a4: 00 00 00 00 00 00    	m.nop
   247aa: 21 a5 00 01  	l.addi.u r20, r20, 0x100
   247ae: 03 8e        	<unknown>
   247b0: 00 00 00 00  	v.nop
   247b4: 00 00 00 00 00 00    	m.nop
   247ba: ef 80 00 00  	l.ldi_lo r16, 0x3
   247be: 23 86        	<unknown>
   247c0: 00 00 00 00  	v.nop
   247c4: 00 00 00 00 00 00    	m.nop
   247ca: ef 80 80 00  	l.ldi_hi r16, 0x3
   247ce: 17 86        	<unknown>
   247d0: 00 00 00 00  	v.nop
   247d4: 00 00 00 00 00 00    	m.nop
   247da: 25 84 d0 02  	l.xorix r16, r16, 0x10
   247de: 1b 86        	<unknown>
   247e0: 00 00 00 00  	v.nop
   247e4: 00 00 00 00 00 00    	m.nop
   247ea: 00 00        	s.nop16
   247ec: 00 00        	s.nop16
   247ee: 1f 86        	<unknown>
   247f0: 00 00 00 00  	v.nop
   247f4: 00 00 00 00 00 00    	m.nop
   247fa: 33 83 00 00  	l.uext.b r12, r16
   247fe: 01 00        	c.nop
   24800: 00 00 00 00  	v.nop
   24804: 00 00 00 00 00 00    	m.nop
   2480a: 00 00        	s.nop16
   2480c: 00 00        	s.nop16
   2480e: 4b 86        	<unknown>
   24810: 00 00 00 00  	v.nop
   24814: 00 00 00 00 00 00    	m.nop
   2481a: bb 04 37 7f  	<unknown>
   2481e: 01 00        	c.nop
   24820: 00 00 00 00  	v.nop
   24824: 00 00 00 00 00 00    	m.nop
   2482a: 71 11 48 22  	l.ld.w r5, [sp, 0x228]
   2482e: 9d 44        	<unknown>
   24830: 00 00 00 00  	v.nop
   24834: 00 00 00 00 00 00    	m.nop
   2483a: 2f 40 00 21  	l.ldi16.s r8, 0x0
   2483e: 01 00        	c.nop
   24840: 00 00 00 00  	v.nop
   24844: 00 00 00 00 00 00    	m.nop
   2484a: 2f 31 80 20  	l.ldi_hi.m1 r6, 0x4
   2484e: 4b 84        	<unknown>
   24850: 00 00 00 00  	v.nop
   24854: 00 00 00 00 00 00    	m.nop
   2485a: ef 39 00 21  	l.ldi16.s r7, 0x7
   2485e: 1b 84        	<unknown>
   24860: 00 00 00 00  	v.nop
   24864: 00 00 00 00 00 00    	m.nop
   2486a: 2f 80 10 21  	l.ldi16.s r16, 0x200
   2486e: 17 84        	<unknown>
   24870: 00 00 00 00  	v.nop
   24874: 00 00 00 00 00 00    	m.nop
   2487a: 23 29 10 03  	l.slli r4, r5, 0x10
   2487e: 1f 84        	<unknown>
   24880: 00 00 00 00  	v.nop
   24884: 00 00 00 00 00 00    	m.nop
   2488a: 6f 30 00 21  	l.ldi16.s r6, 0x1
   2488e: 1a 92        	<unknown>
   24890: 00 00 00 00  	v.nop
   24894: 00 00 00 00 00 00    	m.nop
   2489a: 25 21 05 02  	l.orix r4, r4, 0x5
   2489e: 03 8c        	<unknown>
   248a0: 00 00 00 00  	v.nop
   248a4: 00 00 00 00 00 00    	m.nop
   248aa: af 40 70 1f  	l.ldi16.u r8, 0xfe02
   248ae: 23 84        	<unknown>
   248b0: 00 00 00 00  	v.nop
   248b4: 00 00 00 00 00 00    	m.nop
   248ba: 2f 28 02 21  	l.ldi16.s r5, 0x40
   248be: 01 00        	c.nop
   248c0: 00 00 00 00  	v.nop
   248c4: 00 00 00 00 00 00    	m.nop
   248ca: 6f 30 80 00  	l.ldi_hi r6, 0x1
   248ce: 01 00        	c.nop
   248d0: 00 00 00 00  	v.nop
   248d4: 00 00 00 00 00 00    	m.nop
   248da: ef 39 80 00  	l.ldi_hi r7, 0x7
   248de: 01 00        	c.nop
   248e0: 00 00 00 00  	v.nop
   248e4: 00 00 00 00 00 00    	m.nop
   248ea: af 40 f0 1e  	l.ldi_hi r8, 0xfe02
   248ee: 01 00        	c.nop
   248f0: 00 00 00 00  	v.nop
   248f4: 00 00 00 00 00 00    	m.nop
   248fa: ef 57 ff 3e  	l.ldi_hi.m1 r10, 0xffff
   248fe: 01 00        	c.nop
   24900: 00 00 00 00  	v.nop
   24904: 00 00 00 00 00 00    	m.nop
   2490a: ef 87 ff 1e  	l.ldi_hi r16, 0xffff
   2490e: 01 00        	c.nop
   24910: 00 00 00 00  	v.nop
   24914: 00 00 00 00 00 00    	m.nop
   2491a: 33 4b 00 00  	l.uext.b r12, r9
   2491e: 01 00        	c.nop
   24920: 00 00 00 00  	v.nop
   24924: 00 00 00 00 00 00    	m.nop
   2492a: 00 00        	s.nop16
   2492c: 00 00        	s.nop16
   2492e: 6b 82        	<unknown>
   24930: 00 00 00 00  	v.nop
   24934: 00 00 00 00 00 00    	m.nop
   2493a: 33 53 00 00  	l.uext.b r12, r10
   2493e: 01 00        	c.nop
   24940: 00 00 00 00  	v.nop
   24944: 00 00 00 00 00 00    	m.nop
   2494a: e1 83 40 30  	l.subi.s r15, r16, 0x40
   2494e: 01 00        	c.nop
   24950: 00 00 00 00  	v.nop
   24954: 00 00 00 00 00 00    	m.nop
   2495a: 00 00        	s.nop16
   2495c: 00 00        	s.nop16
   2495e: 27 86        	<unknown>
   24960: 00 00 00 00  	v.nop
   24964: 00 00 00 00 00 00    	m.nop
   2496a: 73 23 00 00  	l.uext.b r13, r4
   2496e: 01 00        	c.nop
   24970: 00 00 00 00  	v.nop
   24974: 00 00 00 00 00 00    	m.nop
   2497a: 00 00        	s.nop16
   2497c: 00 00        	s.nop16
   2497e: c3 56        	<unknown>
   24980: 00 00 00 00  	v.nop
   24984: 00 00 00 00 00 00    	m.nop
   2498a: 73 33 00 00  	l.uext.b r13, r6
   2498e: 01 00        	c.nop
   24990: 00 00 00 00  	v.nop
   24994: 00 00 00 00 00 00    	m.nop
   2499a: 00 00        	s.nop16
   2499c: 00 00        	s.nop16
   2499e: c7 56        	<unknown>
   249a0: 00 00 00 00  	v.nop
   249a4: 00 00 00 00 00 00    	m.nop
   249aa: 73 3b 00 00  	l.uext.b r13, r7
   249ae: 01 00        	c.nop
   249b0: 00 00 00 00  	v.nop
   249b4: 00 00 00 00 00 00    	m.nop
   249ba: b3 43 00 00  	l.uext.b r14, r8
   249be: cb 56        	<unknown>
   249c0: 00 00 00 00  	v.nop
   249c4: 00 00 00 00 00 00    	m.nop
   249ca: 00 00        	s.nop16
   249cc: 00 00        	s.nop16
   249ce: 4f 57        	<unknown>
   249d0: 00 00 00 00  	v.nop
   249d4: 00 00 00 00 00 00    	m.nop
   249da: 2f 74 00 01  	l.ldi16.u r14, 0x10
   249de: 01 00        	c.nop
   249e0: 00 00 00 00  	v.nop
   249e4: 00 00 00 00 00 00    	m.nop
   249ea: 3f 00 81 00  	<unknown>
   249ee: 01 00        	c.nop
   249f0: 30 00 91 90  	<unknown>
   249f4: 00 00 00 00 00 00    	m.nop
   249fa: 21 63 40 00  	l.addi.u r12, r12, 0x40
   249fe: 01 00        	c.nop
   24a00: b0 00 92 90  	<unknown>
   24a04: 00 00 00 00 00 00    	m.nop
   24a0a: 63 73 85 18  	<unknown>
   24a0e: 01 00        	c.nop
   24a10: 30 01 91 90  	<unknown>
   24a14: 00 00 00 00 00 00    	m.nop
   24a1a: 63 6b 9f 0b  	l.srai r13, r13, 0x1f
   24a1e: 01 00        	c.nop
   24a20: b0 01 92 90  	<unknown>
   24a24: 00 00 00 00 00 00    	m.nop
   24a2a: 29 63 8f 18  	<unknown>
   24a2e: 01 00        	c.nop
   24a30: 30 02 91 90  	<unknown>
   24a34: 00 00 00 00 00 00    	m.nop
   24a3a: 00 00        	s.nop16
   24a3c: 00 00        	s.nop16
   24a3e: 01 00        	c.nop
   24a40: b0 02 92 90  	<unknown>
   24a44: 00 00 00 00 00 00    	m.nop
   24a4a: 00 00        	s.nop16
   24a4c: 00 00        	s.nop16
   24a4e: c3 86        	<unknown>
   24a50: 30 03 91 90  	<unknown>
   24a54: 00 00 00 00 00 00    	m.nop
   24a5a: 00 00        	s.nop16
   24a5c: 00 00        	s.nop16
   24a5e: 01 00        	c.nop
   24a60: b0 03 92 90  	<unknown>
   24a64: 00 00 00 00 00 00    	m.nop
   24a6a: 00 00        	s.nop16
   24a6c: 00 00        	s.nop16
   24a6e: 01 00        	c.nop
   24a70: 30 04 91 90  	<unknown>
   24a74: 00 00 00 00 00 00    	m.nop
   24a7a: 00 00        	s.nop16
   24a7c: 00 00        	s.nop16
   24a7e: 01 00        	c.nop
   24a80: b0 04 92 90  	<unknown>
   24a84: 00 00 00 00 00 00    	m.nop
   24a8a: 00 00        	s.nop16
   24a8c: 00 00        	s.nop16
   24a8e: 01 00        	c.nop
   24a90: 30 05 91 90  	<unknown>
   24a94: 00 00 00 00 00 00    	m.nop
   24a9a: 00 00        	s.nop16
   24a9c: 00 00        	s.nop16
   24a9e: 01 00        	c.nop
   24aa0: b0 05 92 90  	<unknown>
   24aa4: 00 00 00 00 00 00    	m.nop
   24aaa: 00 00        	s.nop16
   24aac: 00 00        	s.nop16
   24aae: 01 00        	c.nop
   24ab0: 30 06 91 90  	<unknown>
   24ab4: 00 00 00 00 00 00    	m.nop
   24aba: 00 00        	s.nop16
   24abc: 00 00        	s.nop16
   24abe: 01 00        	c.nop
   24ac0: b0 06 92 90  	<unknown>
   24ac4: 00 00 00 00 00 00    	m.nop
   24aca: 00 00        	s.nop16
   24acc: 00 00        	s.nop16
   24ace: 01 00        	c.nop
   24ad0: 30 07 91 90  	<unknown>
   24ad4: 00 00 00 00 00 00    	m.nop
   24ada: 00 00        	s.nop16
   24adc: 00 00        	s.nop16
   24ade: 01 00        	c.nop
   24ae0: b0 07 92 90  	<unknown>
   24ae4: 00 00 00 00 00 00    	m.nop
   24aea: 00 00        	s.nop16
   24aec: 00 00        	s.nop16
   24aee: 01 00        	c.nop
   24af0: 30 08 91 90  	<unknown>
   24af4: 00 00 00 00 00 00    	m.nop
   24afa: 00 00        	s.nop16
   24afc: 00 00        	s.nop16
   24afe: 01 00        	c.nop
   24b00: b0 08 92 90  	<unknown>
   24b04: 00 00 00 00 00 00    	m.nop
   24b0a: 00 00        	s.nop16
   24b0c: 00 00        	s.nop16
   24b0e: 01 00        	c.nop
   24b10: 30 09 91 90  	<unknown>
   24b14: 00 00 00 00 00 00    	m.nop
   24b1a: 00 00        	s.nop16
   24b1c: 00 00        	s.nop16
   24b1e: 01 00        	c.nop
   24b20: b0 09 92 90  	<unknown>
   24b24: 00 00 00 00 00 00    	m.nop
   24b2a: 00 00        	s.nop16
   24b2c: 00 00        	s.nop16
   24b2e: 01 00        	c.nop
   24b30: 30 0a 91 90  	<unknown>
   24b34: 00 00 00 00 00 00    	m.nop
   24b3a: 00 00        	s.nop16
   24b3c: 00 00        	s.nop16
   24b3e: 01 00        	c.nop
   24b40: b0 0a 92 90  	<unknown>
   24b44: 00 00 00 00 00 00    	m.nop
   24b4a: 00 00        	s.nop16
   24b4c: 00 00        	s.nop16
   24b4e: 01 00        	c.nop
   24b50: 30 0b 91 90  	<unknown>
   24b54: 00 00 00 00 00 00    	m.nop
   24b5a: 00 00        	s.nop16
   24b5c: 00 00        	s.nop16
   24b5e: 01 00        	c.nop
   24b60: b0 0b 92 90  	<unknown>
   24b64: 00 00 00 00 00 00    	m.nop
   24b6a: 00 00        	s.nop16
   24b6c: 00 00        	s.nop16
   24b6e: 01 00        	c.nop
   24b70: 30 0c 91 90  	<unknown>
   24b74: 00 00 00 00 00 00    	m.nop
   24b7a: 00 00        	s.nop16
   24b7c: 00 00        	s.nop16
   24b7e: 01 00        	c.nop
   24b80: b0 0c 92 90  	<unknown>
   24b84: 00 00 00 00 00 00    	m.nop
   24b8a: 00 00        	s.nop16
   24b8c: 00 00        	s.nop16
   24b8e: 01 00        	c.nop
   24b90: 30 0d 91 90  	<unknown>
   24b94: 00 00 00 00 00 00    	m.nop
   24b9a: 00 00        	s.nop16
   24b9c: 00 00        	s.nop16
   24b9e: 01 00        	c.nop
   24ba0: b0 0d 92 90  	<unknown>
   24ba4: 00 00 00 00 00 00    	m.nop
   24baa: 00 00        	s.nop16
   24bac: 00 00        	s.nop16
   24bae: 01 00        	c.nop
   24bb0: 30 0e 91 90  	<unknown>
   24bb4: 00 00 00 00 00 00    	m.nop
   24bba: 00 00        	s.nop16
   24bbc: 00 00        	s.nop16
   24bbe: 01 00        	c.nop
   24bc0: b0 0e 92 90  	<unknown>
   24bc4: 00 00 00 00 00 00    	m.nop
   24bca: 00 00        	s.nop16
   24bcc: 00 00        	s.nop16
   24bce: 01 00        	c.nop
   24bd0: 30 0f 91 90  	<unknown>
   24bd4: 00 00 00 00 00 00    	m.nop
   24bda: 00 00        	s.nop16
   24bdc: 00 00        	s.nop16
   24bde: 41 07        	<unknown>
   24be0: b0 0f 92 90  	<unknown>
   24be4: 00 00 00 00 00 01    	m.nop
   24bea: 2f 74 00 01  	l.ldi16.u r14, 0x10
   24bee: 01 00        	c.nop
   24bf0: 00 00 00 00  	v.nop
   24bf4: 00 00 00 00 00 01    	m.nop
   24bfa: 1e 98        	s.tar.add ta19, ta0
   24bfc: 00 00        	s.nop16
   24bfe: 23 86        	<unknown>
   24c00: 00 00 00 00  	v.nop
   24c04: 00 00 00 00 00 00    	m.nop
   24c0a: 2f 40 00 21  	l.ldi16.s r8, 0x0
   24c0e: 81 44        	<unknown>
   24c10: 00 00 00 00  	v.nop
   24c14: 00 00 00 00 00 00    	m.nop
   24c1a: 00 00        	s.nop16
   24c1c: 00 00        	s.nop16
   24c1e: 03 8c        	<unknown>
   24c20: 00 00 00 00  	v.nop
   24c24: 00 00 00 00 00 00    	m.nop
   24c2a: 39 02 d8 00  	<unknown>
   24c2e: 1b 84        	<unknown>
   24c30: 00 00 00 00  	v.nop
   24c34: 00 00 00 00 00 00    	m.nop
   24c3a: 39 03 c9 00  	<unknown>
   24c3e: 17 84        	<unknown>
   24c40: 00 00 00 00  	v.nop
   24c44: 00 00 00 00 00 00    	m.nop
   24c4a: 39 04 da 00  	<unknown>
   24c4e: 1f 84        	<unknown>
   24c50: 00 00 00 00  	v.nop
   24c54: 00 00 00 00 00 00    	m.nop
   24c5a: 00 00        	s.nop16
   24c5c: 00 00        	s.nop16
   24c5e: 23 84        	<unknown>
   24c60: 00 00 00 00  	v.nop
   24c64: 14 00 02 0c a0 2a    	m.nop
   24c6a: 00 00        	s.nop16
   24c6c: 00 00        	s.nop16
   24c6e: 01 00        	c.nop
   24c70: 00 00 00 00  	v.nop
   24c74: 08 01 00 10 a0 88    	m.nop
   24c7a: 00 00        	s.nop16
   24c7c: 00 00        	s.nop16
   24c7e: 01 00        	c.nop
   24c80: 00 00 00 00  	v.nop
   24c84: 00 00 00 00 00 00    	m.nop
   24c8a: 39 01 c4 4e  	<unknown>
   24c8e: 01 00        	c.nop
   24c90: 00 00 00 00  	v.nop
   24c94: 00 00 00 00 00 00    	m.nop
   24c9a: 3b 25 1a 00  	<unknown>
   24c9e: 01 00        	c.nop
   24ca0: 00 00 00 00  	v.nop
   24ca4: 00 00 00 00 00 00    	m.nop
   24caa: 31 11 5c 14  	l.ld.w r4, [sp, 0x15c]
   24cae: a6 83        	<unknown>
   24cb0: 00 00 00 00  	v.nop
   24cb4: 00 00 00 00 00 00    	m.nop
   24cba: 31 15 58 0c  	l.ld.w r20, [sp, 0xd8]
   24cbe: 62 84        	<unknown>
   24cc0: 00 00 00 00  	v.nop
   24cc4: 00 00 00 00 00 00    	m.nop
   24cca: b1 12 58 14  	l.ld.w r10, [sp, 0x158]
   24cce: 01 00        	c.nop
   24cd0: 00 00 00 00  	v.nop
   24cd4: 00 00 00 00 00 00    	m.nop
   24cda: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   24cde: 01 00        	c.nop
   24ce0: 00 00 00 00  	v.nop
   24ce4: 00 00 00 00 00 00    	m.nop
   24cea: af 2c 80 00  	l.ldi_hi r5, 0x12
   24cee: 01 00        	c.nop
   24cf0: 00 00 00 00  	v.nop
   24cf4: 00 00 00 00 00 00    	m.nop
   24cfa: 23 24 05 18  	<unknown>
   24cfe: a6 82        	<unknown>
   24d00: 00 00 00 00  	v.nop
   24d04: 00 00 00 00 00 00    	m.nop
   24d0a: 00 00        	s.nop16
   24d0c: 00 00        	s.nop16
   24d0e: 42 82        	<unknown>
   24d10: 00 00 00 00  	v.nop
   24d14: 00 00 00 00 00 00    	m.nop
   24d1a: 00 00        	s.nop16
   24d1c: 00 00        	s.nop16
   24d1e: 42 83        	<unknown>
   24d20: 00 00 00 00  	v.nop
   24d24: 00 00 00 00 00 00    	m.nop
   24d2a: 00 00        	s.nop16
   24d2c: 00 00        	s.nop16
   24d2e: d2 84        	<unknown>
   24d30: 00 00 00 00  	v.nop
   24d34: 00 00 00 00 00 00    	m.nop
   24d3a: eb 50 00 01  	l.jalr lr, r10, 0x0
   24d3e: 01 00        	c.nop
   24d40: 00 00 00 00  	v.nop
   24d44: 00 00 00 00 00 00    	m.nop
   24d4a: 71 12 4c 20  	l.ld.w r9, [sp, 0x20c]
   24d4e: 01 44        	<unknown>
   24d50: 00 00 00 00  	v.nop
   24d54: 00 00 00 00 00 00    	m.nop
   24d5a: 00 00        	s.nop16
   24d5c: 00 00        	s.nop16
   24d5e: a2 82        	<unknown>
   24d60: 00 00 00 00  	v.nop
   24d64: 00 00 00 00 00 00    	m.nop
   24d6a: 00 00        	s.nop16
   24d6c: 00 00        	s.nop16
   24d6e: 01 44        	<unknown>
   24d70: 00 00 00 00  	v.nop
   24d74: 00 00 00 00 00 00    	m.nop
   24d7a: 00 00        	s.nop16
   24d7c: 00 00        	s.nop16
   24d7e: a2 83        	<unknown>
   24d80: 00 00 00 00  	v.nop
   24d84: 00 00 00 00 00 00    	m.nop
   24d8a: 00 00        	s.nop16
   24d8c: 00 00        	s.nop16
   24d8e: 42 82        	<unknown>
   24d90: 00 00 00 00  	v.nop
   24d94: 00 00 00 00 00 00    	m.nop
   24d9a: 00 00        	s.nop16
   24d9c: 00 00        	s.nop16
   24d9e: 42 83        	<unknown>
   24da0: 00 00 00 00  	v.nop
   24da4: 00 00 00 00 00 00    	m.nop
   24daa: 00 00        	s.nop16
   24dac: 00 00        	s.nop16
   24dae: 52 84        	<unknown>
   24db0: 00 00 00 00  	v.nop
   24db4: 00 00 00 00 00 00    	m.nop
   24dba: eb 48 00 01  	l.jalr lr, r9, 0x0
   24dbe: 01 00        	c.nop
   24dc0: 00 00 00 00  	v.nop
   24dc4: 00 00 00 00 00 00    	m.nop
   24dca: 00 00        	s.nop16
   24dcc: 00 00        	s.nop16
   24dce: 81 44        	<unknown>
   24dd0: 00 00 00 00  	v.nop
   24dd4: 00 00 00 00 00 00    	m.nop
   24dda: 71 11 58 06  	l.ld.w r5, [sp, 0x78]
   24dde: a6 83        	<unknown>
   24de0: 00 00 00 00  	v.nop
   24de4: 00 00 00 00 00 00    	m.nop
   24dea: 2f 20 08 21  	l.ldi16.s r4, 0x100
   24dee: 01 00        	c.nop
   24df0: 00 00 00 00  	v.nop
   24df4: 00 00 00 00 00 00    	m.nop
   24dfa: 31 12 58 22  	l.ld.w r8, [sp, 0x238]
   24dfe: 01 00        	c.nop
   24e00: 00 00 00 00  	v.nop
   24e04: 00 00 00 00 00 00    	m.nop
   24e0a: b1 12 58 14  	l.ld.w r10, [sp, 0x158]
   24e0e: 01 00        	c.nop
   24e10: 00 00 00 00  	v.nop
   24e14: 00 00 00 00 00 00    	m.nop
   24e1a: 29 25 85 18  	<unknown>
   24e1e: 01 00        	c.nop
   24e20: 00 00 00 00  	v.nop
   24e24: 00 00 00 00 00 00    	m.nop
   24e2a: 31 11 5c 14  	l.ld.w r4, [sp, 0x15c]
   24e2e: 01 00        	c.nop
   24e30: 00 00 00 00  	v.nop
   24e34: 00 00 00 00 00 00    	m.nop
   24e3a: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   24e3e: 01 00        	c.nop
   24e40: 00 00 00 00  	v.nop
   24e44: 00 00 00 00 00 00    	m.nop
   24e4a: af 2c 80 00  	l.ldi_hi r5, 0x12
   24e4e: 01 00        	c.nop
   24e50: 00 00 00 00  	v.nop
   24e54: 00 00 00 00 00 00    	m.nop
   24e5a: 23 24 05 18  	<unknown>
   24e5e: a6 82        	<unknown>
   24e60: 00 00 00 00  	v.nop
   24e64: 00 00 00 00 00 00    	m.nop
   24e6a: 71 12 58 0c  	l.ld.w r9, [sp, 0xd8]
   24e6e: 42 82        	<unknown>
   24e70: 00 00 00 00  	v.nop
   24e74: 00 00 00 00 00 00    	m.nop
   24e7a: 00 00        	s.nop16
   24e7c: 00 00        	s.nop16
   24e7e: 42 83        	<unknown>
   24e80: 00 00 00 00  	v.nop
   24e84: 00 00 00 00 00 00    	m.nop
   24e8a: eb 50 00 01  	l.jalr lr, r10, 0x0
   24e8e: 01 00        	c.nop
   24e90: 00 00 00 00  	v.nop
   24e94: 00 00 00 00 00 00    	m.nop
   24e9a: 7b 93 00 40  	<unknown>
   24e9e: 01 00        	c.nop
   24ea0: 00 00 00 00  	v.nop
   24ea4: 00 00 00 00 00 00    	m.nop
   24eaa: 71 11 54 20  	l.ld.w r5, [sp, 0x214]
   24eae: 01 00        	c.nop
   24eb0: 00 00 00 00  	v.nop
   24eb4: 00 00 00 00 00 00    	m.nop
   24eba: 23 a1 1e 18  	<unknown>
   24ebe: 01 00        	c.nop
   24ec0: 00 00 00 00  	v.nop
   24ec4: 00 00 00 00 00 00    	m.nop
   24eca: bb 2a 84 80  	l.bge.s r5, r4, 0xa <0x24f60>
   24ece: 01 00        	c.nop
   24ed0: 00 00 00 00  	v.nop
   24ed4: 00 00 00 00 00 00    	m.nop
   24eda: 31 11 58 08  	l.ld.w r4, [sp, 0x98]
   24ede: 01 44        	<unknown>
   24ee0: 00 00 00 00  	v.nop
   24ee4: 00 00 00 00 00 00    	m.nop
   24eea: 2f 30 02 21  	l.ldi16.s r6, 0x40
   24eee: a2 82        	<unknown>
   24ef0: 00 00 00 00  	v.nop
   24ef4: 00 00 00 00 00 00    	m.nop
   24efa: 2f 38 08 21  	l.ldi16.s r7, 0x100
   24efe: 66 84        	<unknown>
   24f00: 00 00 00 00  	v.nop
   24f04: 00 00 00 00 00 00    	m.nop
   24f0a: 2f 50 fc 3e  	l.ldi_hi.m1 r10, 0xff80
   24f0e: fa 84        	<unknown>
   24f10: 00 00 00 00  	v.nop
   24f14: 00 00 00 00 00 00    	m.nop
   24f1a: 00 00        	s.nop16
   24f1c: 00 00        	s.nop16
   24f1e: ea 85        	<unknown>
   24f20: 00 00 00 00  	v.nop
   24f24: 00 00 00 00 00 00    	m.nop
   24f2a: 31 10 c4 00  	l.st.w r4, [sp]
   24f2e: 01 00        	c.nop
   24f30: 00 00 00 00  	v.nop
   24f34: 00 00 00 00 00 00    	m.nop
   24f3a: b1 15 4c 0c  	l.ld.w r22, [sp, 0xcc]
   24f3e: 01 00        	c.nop
   24f40: 00 00 00 00  	v.nop
   24f44: 00 00 00 00 00 00    	m.nop
   24f4a: 31 13 5c 1e  	l.ld.w r12, [sp, 0x1fc]
   24f4e: 01 00        	c.nop
   24f50: 00 00 00 00  	v.nop
   24f54: 00 00 00 00 00 00    	m.nop
   24f5a: 2b 30 81 01  	l.jal r0, 0x16 <0x250b0>
   24f5e: 01 00        	c.nop
   24f60: 00 00 00 00  	v.nop
   24f64: 00 00 00 00 00 00    	m.nop
   24f6a: bb 92 16 00  	<unknown>
   24f6e: 01 00        	c.nop
   24f70: 00 00 00 00  	v.nop
   24f74: 00 00 00 00 00 00    	m.nop
   24f7a: b1 15 4c 0c  	l.ld.w r22, [sp, 0xcc]
   24f7e: 01 00        	c.nop
   24f80: 00 00 00 00  	v.nop
   24f84: 00 00 00 00 00 00    	m.nop
   24f8a: 3b 97 1a 40  	<unknown>
   24f8e: 01 00        	c.nop
   24f90: 00 00 00 00  	v.nop
   24f94: 00 00 00 00 00 00    	m.nop
   24f9a: 31 11 5c 10  	l.ld.w r4, [sp, 0x11c]
   24f9e: 01 44        	<unknown>
   24fa0: 00 00 00 00  	v.nop
   24fa4: 00 00 00 00 00 00    	m.nop
   24faa: b1 12 58 08  	l.ld.w r10, [sp, 0x98]
   24fae: a2 82        	<unknown>
   24fb0: 00 00 00 00  	v.nop
   24fb4: 00 00 00 00 00 00    	m.nop
   24fba: 31 13 50 1e  	l.ld.w r12, [sp, 0x1f0]
   24fbe: 66 84        	<unknown>
   24fc0: 00 00 00 00  	v.nop
   24fc4: 00 00 00 00 00 00    	m.nop
   24fca: 2f 30 02 21  	l.ldi16.s r6, 0x40
   24fce: fa 84        	<unknown>
   24fd0: 00 00 00 00  	v.nop
   24fd4: 00 00 00 00 00 00    	m.nop
   24fda: 2f 38 08 21  	l.ldi16.s r7, 0x100
   24fde: 01 00        	c.nop
   24fe0: 00 00 00 00  	v.nop
   24fe4: 00 00 00 00 00 00    	m.nop
   24fea: f1 22 40 00  	l.ld.w r11, [r4]
   24fee: 01 00        	c.nop
   24ff0: 00 00 00 00  	v.nop
   24ff4: 00 00 00 00 00 00    	m.nop
   24ffa: 2b 60 80 01  	l.jal r0, 0xc <0x250b0>
   24ffe: 01 00        	c.nop
   25000: 00 00 00 00  	v.nop
   25004: 00 00 00 00 00 00    	m.nop
   2500a: 31 11 5c 10  	l.ld.w r4, [sp, 0x11c]
   2500e: 81 44        	<unknown>
   25010: 00 00 00 00  	v.nop
   25014: 00 00 00 00 00 00    	m.nop
   2501a: 2f 30 02 21  	l.ldi16.s r6, 0x40
   2501e: a6 82        	<unknown>
   25020: 00 00 00 00  	v.nop
   25024: 00 00 00 00 00 00    	m.nop
   2502a: 2f 38 08 21  	l.ldi16.s r7, 0x100
   2502e: fa 84        	<unknown>
   25030: 00 00 00 00  	v.nop
   25034: 00 00 00 00 00 00    	m.nop
   2503a: 00 00        	s.nop16
   2503c: 00 00        	s.nop16
   2503e: 56 85        	<unknown>
   25040: 00 00 00 00  	v.nop
   25044: 00 00 00 00 00 00    	m.nop
   2504a: 31 22 40 00  	l.ld.w r8, [r4]
   2504e: 01 00        	c.nop
   25050: 00 00 00 00  	v.nop
   25054: 00 00 00 00 00 00    	m.nop
   2505a: 2f 20 fc 3e  	l.ldi_hi.m1 r4, 0xff80
   2505e: 01 00        	c.nop
   25060: 00 00 00 00  	v.nop
   25064: 00 00 00 00 00 00    	m.nop
   2506a: 31 11 c4 00  	l.st.w r4, [sp, 0x4]
   2506e: 01 00        	c.nop
   25070: 00 00 00 00  	v.nop
   25074: 00 00 00 00 00 00    	m.nop
   2507a: b1 15 4c 0c  	l.ld.w r22, [sp, 0xcc]
   2507e: 01 00        	c.nop
   25080: 00 00 00 00  	v.nop
   25084: 00 00 00 00 00 00    	m.nop
   2508a: 31 10 c8 00  	l.st.w r8, [sp]
   2508e: 66 84        	<unknown>
   25090: 00 00 00 00  	v.nop
   25094: 00 00 00 00 00 00    	m.nop
   2509a: f1 12 58 08  	l.ld.w r11, [sp, 0x98]
   2509e: 01 00        	c.nop
   250a0: 00 00 00 00  	v.nop
   250a4: 00 00 00 00 00 00    	m.nop
   250aa: 31 13 54 1e  	l.ld.w r12, [sp, 0x1f4]
   250ae: 01 00        	c.nop
   250b0: 00 00 00 00  	v.nop
   250b4: 00 00 00 00 00 00    	m.nop
   250ba: 00 00        	s.nop16
   250bc: 00 00        	s.nop16
   250be: 5a 82        	<unknown>
   250c0: 00 00 00 00  	v.nop
   250c4: 00 00 00 00 00 00    	m.nop
   250ca: eb 60 00 01  	l.jalr lr, r12, 0x0
   250ce: 01 00        	c.nop
   250d0: 00 00 00 00  	v.nop
   250d4: 00 00 00 00 00 00    	m.nop
   250da: b1 12 58 0c  	l.ld.w r10, [sp, 0xd8]
   250de: 81 44        	<unknown>
   250e0: 00 00 00 00  	v.nop
   250e4: 00 00 00 00 00 00    	m.nop
   250ea: f1 12 5c 20  	l.ld.w r11, [sp, 0x21c]
   250ee: 42 82        	<unknown>
   250f0: 00 00 00 00  	v.nop
   250f4: 00 00 00 00 00 00    	m.nop
   250fa: 00 00        	s.nop16
   250fc: 00 00        	s.nop16
   250fe: a6 82        	<unknown>
   25100: 00 00 00 00  	v.nop
   25104: 00 00 00 00 00 00    	m.nop
   2510a: 00 00        	s.nop16
   2510c: 00 00        	s.nop16
   2510e: 5a 83        	<unknown>
   25110: 00 00 00 00  	v.nop
   25114: 00 00 00 00 00 00    	m.nop
   2511a: 00 00        	s.nop16
   2511c: 00 00        	s.nop16
   2511e: a6 83        	<unknown>
   25120: 00 00 00 00  	v.nop
   25124: 00 00 00 00 00 00    	m.nop
   2512a: 00 00        	s.nop16
   2512c: 00 00        	s.nop16
   2512e: 42 84        	<unknown>
   25130: 00 00 00 00  	v.nop
   25134: 00 00 00 00 00 00    	m.nop
   2513a: eb 58 00 01  	l.jalr lr, r11, 0x0
   2513e: 01 00        	c.nop
   25140: 00 00 00 00  	v.nop
   25144: 00 00 00 00 00 00    	m.nop
   2514a: bb e1 94 c0  	l.blt.s r28, r20, 0x6 <0x251a0>
   2514e: 01 00        	c.nop
   25150: 00 00 00 00  	v.nop
   25154: 00 00 00 00 00 00    	m.nop
   2515a: b1 12 48 20  	l.ld.w r10, [sp, 0x208]
   2515e: 01 44        	<unknown>
   25160: 00 00 00 00  	v.nop
   25164: 00 00 00 00 00 00    	m.nop
   2516a: 2f 30 02 21  	l.ldi16.s r6, 0x40
   2516e: a2 82        	<unknown>
   25170: 00 00 00 00  	v.nop
   25174: 00 00 00 00 00 00    	m.nop
   2517a: 2f 38 08 21  	l.ldi16.s r7, 0x100
   2517e: 42 82        	<unknown>
   25180: 00 00 00 00  	v.nop
   25184: 00 00 00 00 00 00    	m.nop
   2518a: 2f 48 fc 3e  	l.ldi_hi.m1 r9, 0xff80
   2518e: 52 84        	<unknown>
   25190: 00 00 00 00  	v.nop
   25194: 00 00 00 00 00 00    	m.nop
   2519a: eb 50 00 01  	l.jalr lr, r10, 0x0
   2519e: 01 00        	c.nop
   251a0: 00 00 00 00  	v.nop
   251a4: 00 00 00 00 00 00    	m.nop
   251aa: 31 11 54 14  	l.ld.w r4, [sp, 0x154]
   251ae: 81 44        	<unknown>
   251b0: 00 00 00 00  	v.nop
   251b4: 00 00 00 00 00 00    	m.nop
   251ba: 2f 28 08 11  	l.ldi16.u r5, 0x8100
   251be: 01 00        	c.nop
   251c0: 00 00 00 00  	v.nop
   251c4: 00 00 00 00 00 00    	m.nop
   251ca: 2f 2d 80 00  	l.ldi_hi r5, 0x14
   251ce: 01 00        	c.nop
   251d0: 00 00 00 00  	v.nop
   251d4: 00 00 00 00 00 00    	m.nop
   251da: 00 00        	s.nop16
   251dc: 00 00        	s.nop16
   251de: 16 92        	<unknown>
   251e0: 00 00 00 00  	v.nop
   251e4: 00 00 00 00 00 00    	m.nop
   251ea: 00 00        	s.nop16
   251ec: 00 00        	s.nop16
   251ee: 12 8a        	<unknown>
   251f0: 00 00 00 00  	v.nop
   251f4: 00 00 00 00 00 00    	m.nop
   251fa: 7b f1 09 00  	<unknown>
   251fe: 01 00        	c.nop
   25200: 00 00 00 00  	v.nop
   25204: 00 00 00 00 00 00    	m.nop
   2520a: 71 11 44 20  	l.ld.w r5, [sp, 0x204]
   2520e: 01 00        	c.nop
   25210: 00 00 00 00  	v.nop
   25214: 00 00 00 00 00 00    	m.nop
   2521a: 2f 30 00 11  	l.ldi16.u r6, 0x8000
   2521e: 01 00        	c.nop
   25220: 00 00 00 00  	v.nop
   25224: 00 00 00 00 00 00    	m.nop
   2522a: 2f 35 80 00  	l.ldi_hi r6, 0x14
   2522e: 01 00        	c.nop
   25230: 00 00 00 00  	v.nop
   25234: 00 00 00 00 00 00    	m.nop
   2523a: 23 2d 06 18  	<unknown>
   2523e: 01 00        	c.nop
   25240: 00 00 00 00  	v.nop
   25244: 00 00 00 00 00 00    	m.nop
   2524a: 71 11 50 14  	l.ld.w r5, [sp, 0x150]
   2524e: 01 00        	c.nop
   25250: 00 00 00 00  	v.nop
   25254: 00 00 00 00 00 00    	m.nop
   2525a: 2f 30 78 2f  	l.ldi16.s r6, 0x7f00
   2525e: 01 00        	c.nop
   25260: 00 00 00 00  	v.nop
   25264: 00 00 00 00 00 00    	m.nop
   2526a: 2f 35 80 00  	l.ldi_hi r6, 0x14
   2526e: 01 00        	c.nop
   25270: 00 00 00 00  	v.nop
   25274: 00 00 00 00 00 00    	m.nop
   2527a: 00 00        	s.nop16
   2527c: 00 00        	s.nop16
   2527e: 9a 92        	<unknown>
   25280: 00 00 00 00  	v.nop
   25284: 00 00 00 00 00 00    	m.nop
   2528a: 00 00        	s.nop16
   2528c: 00 00        	s.nop16
   2528e: 96 8b        	<unknown>
   25290: 00 00 00 00  	v.nop
   25294: 00 00 00 00 00 00    	m.nop
   2529a: 3b f5 69 00  	<unknown>
   2529e: 01 00        	c.nop
   252a0: 00 00 00 00  	v.nop
   252a4: 00 00 00 00 00 00    	m.nop
   252aa: b1 11 40 20  	l.ld.w r6, [sp, 0x200]
   252ae: 01 00        	c.nop
   252b0: 00 00 00 00  	v.nop
   252b4: 00 00 00 00 00 00    	m.nop
   252ba: 2f 38 70 2f  	l.ldi16.s r7, 0x7e00
   252be: 01 00        	c.nop
   252c0: 00 00 00 00  	v.nop
   252c4: 00 00 00 00 00 00    	m.nop
   252ca: 2f 3d 80 00  	l.ldi_hi r7, 0x14
   252ce: 01 00        	c.nop
   252d0: 00 00 00 00  	v.nop
   252d4: 00 00 00 00 00 00    	m.nop
   252da: e3 35 07 18  	<unknown>
   252de: 01 00        	c.nop
   252e0: 00 00 00 00  	v.nop
   252e4: 00 00 00 00 00 00    	m.nop
   252ea: 71 16 54 22  	l.ld.w r25, [sp, 0x234]
   252ee: 01 00        	c.nop
   252f0: 00 00 00 00  	v.nop
   252f4: 00 00 00 00 00 00    	m.nop
   252fa: 3b f4 69 40  	<unknown>
   252fe: 01 00        	c.nop
   25300: 00 00 00 00  	v.nop
   25304: 00 00 00 00 00 00    	m.nop
   2530a: b1 15 50 22  	l.ld.w r22, [sp, 0x230]
   2530e: 01 00        	c.nop
   25310: 00 00 00 00  	v.nop
   25314: 00 00 00 00 00 00    	m.nop
   2531a: bb f0 09 00  	<unknown>
   2531e: 01 00        	c.nop
   25320: 00 00 00 00  	v.nop
   25324: 00 00 00 00 00 00    	m.nop
   2532a: 00 00        	s.nop16
   2532c: 00 00        	s.nop16
   2532e: 16 8b        	<unknown>
   25330: 00 00 00 00  	v.nop
   25334: 00 00 00 00 00 00    	m.nop
   2533a: 2f 20 fc 26  	l.ldi_hi.m1 r4, 0x3f80
   2533e: a6 82        	<unknown>
   25340: 00 00 00 00  	v.nop
   25344: 00 00 00 00 00 00    	m.nop
   2534a: 2f 30 08 21  	l.ldi16.s r6, 0x100
   2534e: a6 83        	<unknown>
   25350: 00 00 00 00  	v.nop
   25354: 00 00 00 00 00 00    	m.nop
   2535a: 2f 40 02 21  	l.ldi16.s r8, 0x40
   2535e: 52 85        	<unknown>
   25360: 00 00 00 00  	v.nop
   25364: 00 00 00 00 00 00    	m.nop
   2536a: 31 14 c4 00  	l.st.w r4, [sp, 0x10]
   2536e: 42 82        	<unknown>
   25370: 00 00 00 00  	v.nop
   25374: 00 00 00 00 00 00    	m.nop
   2537a: 31 13 c6 00  	l.st.w r6, [sp, 0xc]
   2537e: 42 83        	<unknown>
   25380: 00 00 00 00  	v.nop
   25384: 00 00 00 00 00 00    	m.nop
   2538a: 31 12 c8 00  	l.st.w r8, [sp, 0x8]
   2538e: 66 84        	<unknown>
   25390: 00 00 00 00  	v.nop
   25394: 00 00 00 00 00 00    	m.nop
   2539a: 31 11 c9 00  	l.st.w r9, [sp, 0x4]
   2539e: a6 85        	<unknown>
   253a0: 00 00 00 00  	v.nop
   253a4: 00 00 00 00 00 00    	m.nop
   253aa: 31 10 d7 00  	l.st.w r23, [sp]
   253ae: 01 00        	c.nop
   253b0: 00 00 00 00  	v.nop
   253b4: 00 00 00 00 00 00    	m.nop
   253ba: 31 13 44 22  	l.ld.w r12, [sp, 0x224]
   253be: 01 00        	c.nop
   253c0: 00 00 00 00  	v.nop
   253c4: 00 00 00 00 00 00    	m.nop
   253ca: eb 60 00 01  	l.jalr lr, r12, 0x0
   253ce: 01 00        	c.nop
   253d0: 00 00 00 00  	v.nop
   253d4: 00 00 00 00 00 00    	m.nop
   253da: b1 12 5c 08  	l.ld.w r10, [sp, 0x9c]
   253de: 01 00        	c.nop
   253e0: 00 00 00 00  	v.nop
   253e4: 00 00 00 00 00 00    	m.nop
   253ea: 7b db 40 00  	<unknown>
   253ee: 01 00        	c.nop
   253f0: 00 00 00 00  	v.nop
   253f4: 00 00 00 00 00 00    	m.nop
   253fa: 31 11 58 20  	l.ld.w r4, [sp, 0x218]
   253fe: 01 00        	c.nop
   25400: 00 00 00 00  	v.nop
   25404: 00 00 00 00 00 00    	m.nop
   2540a: fb 22 de 80  	l.bge.s r4, r30, 0x4b <0x258b0>
   2540e: 01 00        	c.nop
   25410: 00 00 00 00  	v.nop
   25414: 00 00 00 00 00 00    	m.nop
   2541a: 31 11 54 0a  	l.ld.w r4, [sp, 0xb4]
   2541e: 81 44        	<unknown>
   25420: 00 00 00 00  	v.nop
   25424: 00 00 00 00 00 00    	m.nop
   2542a: 2f 28 02 21  	l.ldi16.s r5, 0x40
   2542e: 01 00        	c.nop
   25430: 00 00 00 00  	v.nop
   25434: 00 00 00 00 00 00    	m.nop
   2543a: 00 00        	s.nop16
   2543c: 00 00        	s.nop16
   2543e: 7a 92        	<unknown>
   25440: 00 00 00 00  	v.nop
   25444: 00 00 00 00 00 00    	m.nop
   2544a: 69 29 84 18  	<unknown>
   2544e: 71 42        	<unknown>
   25450: 00 00 00 00  	v.nop
   25454: 00 00 00 00 00 00    	m.nop
   2545a: 23 51 04 10  	<unknown>
   2545e: 01 00        	c.nop
   25460: 00 00 00 00  	v.nop
   25464: 00 00 00 00 00 00    	m.nop
   2546a: a3 29 09 03  	l.slli r6, r5, 0x9
   2546e: 01 00        	c.nop
   25470: 00 00 00 00  	v.nop
   25474: 00 00 00 00 00 00    	m.nop
   2547a: 31 20 c9 00  	l.st.w r9, [r4]
   2547e: 01 00        	c.nop
   25480: 00 00 00 00  	v.nop
   25484: 00 00 00 00 00 00    	m.nop
   2548a: 31 27 c9 fe  	l.st.w r9, [r4, -0x4]
   2548e: 01 00        	c.nop
   25490: 00 00 00 00  	v.nop
   25494: 00 00 00 00 00 00    	m.nop
   2549a: 31 22 d0 00  	l.st.w r16, [r4, 0x8]
   2549e: 01 00        	c.nop
   254a0: 00 00 00 00  	v.nop
   254a4: 00 00 00 00 00 00    	m.nop
   254aa: 31 21 c9 00  	l.st.w r9, [r4, 0x4]
   254ae: 01 00        	c.nop
   254b0: 00 00 00 00  	v.nop
   254b4: 00 00 00 00 00 00    	m.nop
   254ba: 31 25 c6 00  	l.st.w r6, [r4, 0x14]
   254be: 01 00        	c.nop
   254c0: 00 00 00 00  	v.nop
   254c4: 00 00 00 00 00 00    	m.nop
   254ca: 2f 32 10 21  	l.ldi16.s r6, 0x208
   254ce: 01 00        	c.nop
   254d0: 00 00 00 00  	v.nop
   254d4: 00 00 00 00 00 00    	m.nop
   254da: 2f 30 81 00  	l.ldi_hi r6, 0x20
   254de: 01 00        	c.nop
   254e0: 00 00 00 00  	v.nop
   254e4: 00 00 00 00 00 00    	m.nop
   254ea: a1 31 00 2f  	l.addi.s r6, r6, -0x100
   254ee: 01 00        	c.nop
   254f0: 00 00 00 00  	v.nop
   254f4: 00 00 00 00 00 00    	m.nop
   254fa: 31 27 c9 0a  	l.st.w r9, [r4, 0xbc]
   254fe: 01 00        	c.nop
   25500: 00 00 00 00  	v.nop
   25504: 00 00 00 00 00 00    	m.nop
   2550a: 31 26 c6 fe  	l.st.w r6, [r4, -0x8]
   2550e: 11 43        	<unknown>
   25510: 00 00 00 00  	v.nop
   25514: 00 00 00 00 00 00    	m.nop
   2551a: 33 b9 83 00  	l.movs r4, md23_wrk
   2551e: 01 00        	c.nop
   25520: 00 00 00 00  	v.nop
   25524: 00 00 00 00 00 00    	m.nop
   2552a: 25 21 86 01  	l.andix r4, r4, 0x6
   2552e: 01 00        	c.nop
   25530: 00 00 00 00  	v.nop
   25534: 00 00 00 00 00 00    	m.nop
   2553a: 7b 24 00 00  	<unknown>
   2553e: 01 00        	c.nop
   25540: 00 00 00 00  	v.nop
   25544: 00 00 00 00 00 00    	m.nop
   2554a: 2f 22 04 21  	l.ldi16.s r4, 0x88
   2554e: 01 00        	c.nop
   25550: 00 00 00 00  	v.nop
   25554: 00 00 00 00 00 00    	m.nop
   2555a: 23 51 04 10  	<unknown>
   2555e: 01 00        	c.nop
   25560: 00 00 00 00  	v.nop
   25564: 00 00 00 00 00 00    	m.nop
   2556a: b1 21 40 00  	l.ld.w r6, [r4]
   2556e: 01 00        	c.nop
   25570: 00 00 00 00  	v.nop
   25574: 00 00 00 00 00 00    	m.nop
   2557a: f3 a1 83 00  	l.movs r7, md20_wrk
   2557e: 01 00        	c.nop
   25580: 00 00 00 00  	v.nop
   25584: 00 00 00 00 00 00    	m.nop
   2558a: 2f 58 f0 3e  	l.ldi_hi.m1 r11, 0xfe00
   2558e: 86 03        	<unknown>
   25590: 00 00 00 00  	v.nop
   25594: 00 00 00 00 00 00    	m.nop
   2559a: e5 39 8b 01  	l.andix r7, r7, 0xb
   2559e: 01 00        	c.nop
   255a0: 00 00 00 00  	v.nop
   255a4: 00 00 00 00 00 00    	m.nop
   255aa: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   255ae: 01 00        	c.nop
   255b0: 00 00 00 00  	v.nop
   255b4: 00 00 00 00 00 00    	m.nop
   255ba: 33 02 80 00  	l.movs r8, sip_id
   255be: 01 00        	c.nop
   255c0: 00 00 00 00  	v.nop
   255c4: 00 00 00 00 00 00    	m.nop
   255ca: 6f 58 80 00  	l.ldi_hi r11, 0x1
   255ce: 3e 04        	<unknown>
   255d0: 00 00 00 00  	v.nop
   255d4: 00 00 00 00 00 00    	m.nop
   255da: a5 31 8b 01  	l.andix r6, r6, 0xb
   255de: 01 00        	c.nop
   255e0: 00 00 00 00  	v.nop
   255e4: 00 00 00 00 00 00    	m.nop
   255ea: f1 12 4c 02  	l.ld.w r11, [sp, 0x2c]
   255ee: 01 00        	c.nop
   255f0: 00 00 00 00  	v.nop
   255f4: 00 00 00 00 00 00    	m.nop
   255fa: a5 31 07 02  	l.orix r6, r6, 0x7
   255fe: 01 00        	c.nop
   25600: 00 00 00 00  	v.nop
   25604: 00 00 00 00 00 00    	m.nop
   2560a: 00 00        	s.nop16
   2560c: 00 00        	s.nop16
   2560e: 6d 8c        	<unknown>
   25610: 00 00 00 00  	v.nop
   25614: 00 00 00 00 00 00    	m.nop
   2561a: a5 31 08 02  	l.orix r6, r6, 0x8
   2561e: 01 00        	c.nop
   25620: 00 00 00 00  	v.nop
   25624: 00 00 00 00 00 00    	m.nop
   2562a: a5 31 1d 02  	l.orix r6, r6, 0x1d
   2562e: 01 00        	c.nop
   25630: 00 00 00 00  	v.nop
   25634: 00 00 00 00 00 00    	m.nop
   2563a: 31 20 c6 00  	l.st.w r6, [r4]
   2563e: 01 00        	c.nop
   25640: 00 00 00 00  	v.nop
   25644: 00 00 00 00 00 00    	m.nop
   2564a: 2f 25 01 21  	l.ldi16.s r4, 0x34
   2564e: 8a 02        	<unknown>
   25650: 00 00 00 00  	v.nop
   25654: 00 00 00 00 00 00    	m.nop
   2565a: 23 51 04 10  	<unknown>
   2565e: 01 00        	c.nop
   25660: 00 00 00 00  	v.nop
   25664: 00 00 00 00 00 00    	m.nop
   2566a: b3 01 80 00  	l.movs r6, sip_id
   2566e: 01 00        	c.nop
   25670: 00 00 00 00  	v.nop
   25674: 00 00 00 00 00 00    	m.nop
   2567a: 31 20 da 00  	l.st.w r26, [r4]
   2567e: 01 00        	c.nop
   25680: 00 00 00 00  	v.nop
   25684: 00 00 00 00 00 00    	m.nop
   2568a: b1 11 4c 06  	l.ld.w r6, [sp, 0x6c]
   2568e: 01 00        	c.nop
   25690: 00 00 00 00  	v.nop
   25694: 00 00 00 00 00 00    	m.nop
   2569a: 2d 30 20 00  	l.wait.dec r6
   2569e: 01 00        	c.nop
   256a0: 00 00 00 00  	v.nop
   256a4: 00 00 00 00 00 00    	m.nop
   256aa: 31 22 c9 fe  	l.st.w r9, [r4, -0x18]
   256ae: 11 43        	<unknown>
   256b0: 00 00 00 00  	v.nop
   256b4: 00 00 00 00 00 00    	m.nop
   256ba: 31 21 c9 fe  	l.st.w r9, [r4, -0x1c]
   256be: 01 00        	c.nop
   256c0: 00 00 00 00  	v.nop
   256c4: 00 00 00 00 00 00    	m.nop
   256ca: 31 24 d7 fe  	l.st.w r23, [r4, -0x10]
   256ce: 01 00        	c.nop
   256d0: 00 00 00 00  	v.nop
   256d4: 00 00 00 00 00 00    	m.nop
   256da: 31 23 c9 fe  	l.st.w r9, [r4, -0x14]
   256de: 01 00        	c.nop
   256e0: 00 00 00 00  	v.nop
   256e4: 00 00 00 00 00 00    	m.nop
   256ea: 31 27 c5 fe  	l.st.w r5, [r4, -0x4]
   256ee: 01 00        	c.nop
   256f0: 00 00 00 00  	v.nop
   256f4: 00 00 00 00 00 00    	m.nop
   256fa: 2f 28 fc 26  	l.ldi_hi.m1 r5, 0x3f80
   256fe: 01 00        	c.nop
   25700: 00 00 00 00  	v.nop
   25704: 00 00 00 00 00 00    	m.nop
   2570a: 31 21 c5 0a  	l.st.w r5, [r4, 0xa4]
   2570e: 01 00        	c.nop
   25710: 00 00 00 00  	v.nop
   25714: 00 00 00 00 00 00    	m.nop
   2571a: 2f 2a 10 21  	l.ldi16.s r5, 0x208
   2571e: 01 00        	c.nop
   25720: 00 00 00 00  	v.nop
   25724: 00 00 00 00 00 00    	m.nop
   2572a: 2f 28 81 00  	l.ldi_hi r5, 0x20
   2572e: 01 00        	c.nop
   25730: 00 00 00 00  	v.nop
   25734: 00 00 00 00 00 00    	m.nop
   2573a: 31 20 c5 fe  	l.st.w r5, [r4, -0x20]
   2573e: 01 00        	c.nop
   25740: 00 00 00 00  	v.nop
   25744: 00 00 00 00 00 00    	m.nop
   2574a: 73 b9 83 00  	l.movs r5, md23_wrk
   2574e: 01 00        	c.nop
   25750: 00 00 00 00  	v.nop
   25754: 00 00 00 00 00 00    	m.nop
   2575a: 65 29 86 01  	l.andix r5, r5, 0x6
   2575e: 01 00        	c.nop
   25760: 00 00 00 00  	v.nop
   25764: 00 00 00 00 00 00    	m.nop
   2576a: 7b 2c 00 00  	<unknown>
   2576e: 01 00        	c.nop
   25770: 00 00 00 00  	v.nop
   25774: 00 00 00 00 00 00    	m.nop
   2577a: 2f 2a 04 21  	l.ldi16.s r5, 0x88
   2577e: 01 00        	c.nop
   25780: 00 00 00 00  	v.nop
   25784: 00 00 00 00 00 00    	m.nop
   2578a: 63 51 05 10  	<unknown>
   2578e: 01 00        	c.nop
   25790: 00 00 00 00  	v.nop
   25794: 00 00 00 00 00 00    	m.nop
   2579a: b1 29 40 00  	l.ld.w r6, [r5]
   2579e: 01 00        	c.nop
   257a0: 00 00 00 00  	v.nop
   257a4: 00 00 00 00 00 00    	m.nop
   257aa: f3 a1 83 00  	l.movs r7, md20_wrk
   257ae: 01 00        	c.nop
   257b0: 00 00 00 00  	v.nop
   257b4: 00 00 00 00 00 00    	m.nop
   257ba: 2f 48 f0 3e  	l.ldi_hi.m1 r9, 0xfe00
   257be: 86 03        	<unknown>
   257c0: 00 00 00 00  	v.nop
   257c4: 00 00 00 00 00 00    	m.nop
   257ca: e5 39 89 01  	l.andix r7, r7, 0x9
   257ce: 01 00        	c.nop
   257d0: 00 00 00 00  	v.nop
   257d4: 00 00 00 00 00 00    	m.nop
   257da: ef 4f 7f 1f  	l.ldi16.u r9, 0xffff
   257de: 01 00        	c.nop
   257e0: 00 00 00 00  	v.nop
   257e4: 00 00 00 00 00 00    	m.nop
   257ea: 33 02 80 00  	l.movs r8, sip_id
   257ee: 01 00        	c.nop
   257f0: 00 00 00 00  	v.nop
   257f4: 00 00 00 00 00 00    	m.nop
   257fa: 6f 48 80 00  	l.ldi_hi r9, 0x1
   257fe: 3e 04        	<unknown>
   25800: 00 00 00 00  	v.nop
   25804: 00 00 00 00 00 00    	m.nop
   2580a: a5 31 89 01  	l.andix r6, r6, 0x9
   2580e: 01 00        	c.nop
   25810: 00 00 00 00  	v.nop
   25814: 00 00 00 00 00 00    	m.nop
   2581a: 71 12 4c 02  	l.ld.w r9, [sp, 0x2c]
   2581e: 01 00        	c.nop
   25820: 00 00 00 00  	v.nop
   25824: 00 00 00 00 00 00    	m.nop
   2582a: a5 31 07 02  	l.orix r6, r6, 0x7
   2582e: 01 00        	c.nop
   25830: 00 00 00 00  	v.nop
   25834: 00 00 00 00 00 00    	m.nop
   2583a: 00 00        	s.nop16
   2583c: 00 00        	s.nop16
   2583e: 65 8c        	<unknown>
   25840: 00 00 00 00  	v.nop
   25844: 00 00 00 00 00 00    	m.nop
   2584a: a5 31 08 02  	l.orix r6, r6, 0x8
   2584e: 01 00        	c.nop
   25850: 00 00 00 00  	v.nop
   25854: 00 00 00 00 00 00    	m.nop
   2585a: a5 31 1d 02  	l.orix r6, r6, 0x1d
   2585e: 01 00        	c.nop
   25860: 00 00 00 00  	v.nop
   25864: 00 00 00 00 00 00    	m.nop
   2586a: 31 28 c6 00  	l.st.w r6, [r5]
   2586e: 01 00        	c.nop
   25870: 00 00 00 00  	v.nop
   25874: 00 00 00 00 00 00    	m.nop
   2587a: 73 01 80 00  	l.movs r5, sip_id
   2587e: 01 00        	c.nop
   25880: 00 00 00 00  	v.nop
   25884: 00 00 00 00 00 00    	m.nop
   2588a: 31 20 da 00  	l.st.w r26, [r4]
   2588e: 01 00        	c.nop
   25890: 00 00 00 00  	v.nop
   25894: 00 00 00 00 00 00    	m.nop
   2589a: 31 11 4c 06  	l.ld.w r4, [sp, 0x6c]
   2589e: 01 00        	c.nop
   258a0: 00 00 00 00  	v.nop
   258a4: 00 00 00 00 00 00    	m.nop
   258aa: 2d 20 20 00  	l.wait.dec r4
   258ae: 01 00        	c.nop
   258b0: 00 00 00 00  	v.nop
   258b4: 00 00 00 00 00 00    	m.nop
   258ba: 31 11 54 06  	l.ld.w r4, [sp, 0x74]
   258be: 01 00        	c.nop
   258c0: 00 00 00 00  	v.nop
   258c4: 00 00 00 00 00 00    	m.nop
   258ca: 2f 2a 10 21  	l.ldi16.s r5, 0x208
   258ce: 01 00        	c.nop
   258d0: 00 00 00 00  	v.nop
   258d4: 00 00 00 00 00 00    	m.nop
   258da: 00 00        	s.nop16
   258dc: 00 00        	s.nop16
   258de: 4e 92        	<unknown>
   258e0: 00 00 00 00  	v.nop
   258e4: 00 00 00 00 00 00    	m.nop
   258ea: 31 21 40 00  	l.ld.w r4, [r4]
   258ee: 01 00        	c.nop
   258f0: 00 00 00 00  	v.nop
   258f4: 00 00 00 00 00 00    	m.nop
   258fa: 00 00        	s.nop16
   258fc: 00 00        	s.nop16
   258fe: 16 92        	<unknown>
   25900: 00 00 00 00  	v.nop
   25904: 00 00 00 00 00 00    	m.nop
   2590a: 71 21 10 04  	l.ld.b r5, [r4, 0x50]
   2590e: 01 00        	c.nop
   25910: 00 00 00 00  	v.nop
   25914: 00 00 00 00 00 00    	m.nop
   2591a: 65 29 9c 01  	l.andix r5, r5, 0x1c
   2591e: 01 00        	c.nop
   25920: 00 00 00 00  	v.nop
   25924: 00 00 00 00 00 00    	m.nop
   2592a: 7b 2c 00 00  	<unknown>
   2592e: 01 00        	c.nop
   25930: 00 00 00 00  	v.nop
   25934: 00 00 00 00 00 00    	m.nop
   2593a: b1 21 4c 04  	l.ld.w r6, [r4, 0x4c]
   2593e: 01 00        	c.nop
   25940: 00 00 00 00  	v.nop
   25944: 00 00 00 00 00 00    	m.nop
   2594a: f1 21 48 04  	l.ld.w r7, [r4, 0x48]
   2594e: 01 00        	c.nop
   25950: 00 00 00 00  	v.nop
   25954: 00 00 00 00 00 00    	m.nop
   2595a: 61 21 4c 00  	l.addi.u r5, r4, 0x4c
   2595e: 01 00        	c.nop
   25960: 00 00 00 00  	v.nop
   25964: 00 00 00 00 00 00    	m.nop
   2596a: fb 34 07 40  	<unknown>
   2596e: 01 00        	c.nop
   25970: 00 00 00 00  	v.nop
   25974: 00 00 00 00 00 00    	m.nop
   2597a: f1 21 4c 02  	l.ld.w r7, [r4, 0x2c]
   2597e: 01 43        	<unknown>
   25980: 00 00 00 00  	v.nop
   25984: 00 00 00 00 00 00    	m.nop
   2598a: 31 22 48 02  	l.ld.w r8, [r4, 0x28]
   2598e: 81 44        	<unknown>
   25990: 00 00 00 00  	v.nop
   25994: 00 00 00 00 00 00    	m.nop
   2599a: 31 23 c0 04  	l.st.w r0, [r4, 0x4c]
   2599e: 01 00        	c.nop
   259a0: 00 00 00 00  	v.nop
   259a4: 00 00 00 00 00 00    	m.nop
   259aa: 00 00        	s.nop16
   259ac: 00 00        	s.nop16
   259ae: a2 93        	<unknown>
   259b0: 00 00 00 00  	v.nop
   259b4: 00 00 00 00 00 00    	m.nop
   259ba: 31 23 c7 02  	l.st.w r7, [r4, 0x2c]
   259be: 01 00        	c.nop
   259c0: 00 00 00 00  	v.nop
   259c4: 00 00 00 00 00 00    	m.nop
   259ca: 2b 70 80 01  	l.jal r0, 0xe <0x25aa0>
   259ce: 01 00        	c.nop
   259d0: 00 00 00 00  	v.nop
   259d4: 00 00 00 00 00 00    	m.nop
   259da: 71 16 54 22  	l.ld.w r25, [sp, 0x234]
   259de: 01 00        	c.nop
   259e0: 00 00 00 00  	v.nop
   259e4: 00 00 00 00 00 00    	m.nop
   259ea: bb f4 09 3f  	<unknown>
   259ee: 01 00        	c.nop
   259f0: 00 00 00 00  	v.nop
   259f4: 00 00 00 00 00 00    	m.nop
   259fa: 00 00        	s.nop16
   259fc: 00 00        	s.nop16
   259fe: 92 8c        	<unknown>
   25a00: 00 00 00 00  	v.nop
   25a04: 00 00 00 00 00 00    	m.nop
   25a0a: b1 15 50 22  	l.ld.w r22, [sp, 0x230]
   25a0e: 01 00        	c.nop
   25a10: 00 00 00 00  	v.nop
   25a14: 00 00 00 00 00 00    	m.nop
   25a1a: 7b f4 09 7f  	<unknown>
   25a1e: 01 00        	c.nop
   25a20: 00 00 00 00  	v.nop
   25a24: 00 00 00 00 00 00    	m.nop
   25a2a: 2b 08 f9 3f  	l.jal r0, -0x6f <0x25330>
   25a2e: 01 00        	c.nop
   25a30: 00 00 00 00  	v.nop
   25a34: 00 00 00 00 00 00    	m.nop
   25a3a: 71 21 10 02  	l.ld.b r5, [r4, 0x30]
   25a3e: 81 44        	<unknown>
   25a40: 00 00 00 00  	v.nop
   25a44: 00 00 00 00 00 00    	m.nop
   25a4a: 65 29 9c 01  	l.andix r5, r5, 0x1c
   25a4e: 01 00        	c.nop
   25a50: 00 00 00 00  	v.nop
   25a54: 00 00 00 00 00 00    	m.nop
   25a5a: 3b 29 20 00  	<unknown>
   25a5e: 01 00        	c.nop
   25a60: 00 00 00 00  	v.nop
   25a64: 00 00 00 00 00 00    	m.nop
   25a6a: 61 21 2c 00  	l.addi.u r5, r4, 0x2c
   25a6e: 01 00        	c.nop
   25a70: 00 00 00 00  	v.nop
   25a74: 00 00 00 00 00 00    	m.nop
   25a7a: b1 21 4c 02  	l.ld.w r6, [r4, 0x2c]
   25a7e: 01 00        	c.nop
   25a80: 00 00 00 00  	v.nop
   25a84: 00 00 00 00 00 00    	m.nop
   25a8a: 2b 10 80 01  	l.jal r0, 0x2 <0x25aa0>
   25a8e: 01 00        	c.nop
   25a90: 00 00 00 00  	v.nop
   25a94: 00 00 00 00 00 00    	m.nop
   25a9a: 00 00        	s.nop16
   25a9c: 00 00        	s.nop16
   25a9e: 81 44        	<unknown>
   25aa0: 00 00 00 00  	v.nop
   25aa4: 00 00 00 00 00 00    	m.nop
   25aaa: 00 00        	s.nop16
   25aac: 00 00        	s.nop16
   25aae: 05 03        	<unknown>
   25ab0: 00 00 00 00  	v.nop
   25ab4: 00 00 00 00 00 00    	m.nop
   25aba: 31 28 c6 00  	l.st.w r6, [r5]
   25abe: 01 00        	c.nop
   25ac0: 00 00 00 00  	v.nop
   25ac4: 00 00 00 00 00 00    	m.nop
   25aca: 73 01 80 00  	l.movs r5, sip_id
   25ace: 01 00        	c.nop
   25ad0: 00 00 00 00  	v.nop
   25ad4: 00 00 00 00 00 00    	m.nop
   25ada: b3 79 83 00  	l.movs r6, md15_wrk
   25ade: c2 02        	<unknown>
   25ae0: 00 00 00 00  	v.nop
   25ae4: 00 00 00 00 00 00    	m.nop
   25aea: a3 31 90 03  	l.srli r6, r6, 0x10
   25aee: 1f f0        	<unknown>
   25af0: 00 00 00 00  	v.nop
   25af4: 00 00 00 00 00 00    	m.nop
   25afa: a5 31 9c 01  	l.andix r6, r6, 0x1c
   25afe: 01 00        	c.nop
   25b00: 00 00 00 00  	v.nop
   25b04: 00 00 00 00 00 00    	m.nop
   25b0a: f1 22 4c 02  	l.ld.w r11, [r4, 0x2c]
   25b0e: 22 93        	<unknown>
   25b10: 00 00 00 00  	v.nop
   25b14: 00 00 00 00 00 00    	m.nop
   25b1a: ef 47 80 20  	l.ldi_hi.m1 r8, 0x1f
   25b1e: 12 03        	<unknown>
   25b20: 00 00 00 00  	v.nop
   25b24: 00 00 00 00 00 00    	m.nop
   25b2a: 65 29 88 01  	l.andix r5, r5, 0x8
   25b2e: 01 00        	c.nop
   25b30: 00 00 00 00  	v.nop
   25b34: 00 00 00 00 00 00    	m.nop
   25b3a: ef 47 7f 1f  	l.ldi16.u r8, 0xffff
   25b3e: 01 00        	c.nop
   25b40: 00 00 00 00  	v.nop
   25b44: 00 00 00 00 00 00    	m.nop
   25b4a: ef 47 83 00  	l.ldi_hi r8, 0x7f
   25b4e: 01 00        	c.nop
   25b50: 00 00 00 00  	v.nop
   25b54: 00 00 00 00 00 00    	m.nop
   25b5a: f1 21 40 02  	l.ld.w r7, [r4, 0x20]
   25b5e: 22 86        	<unknown>
   25b60: 00 00 00 00  	v.nop
   25b64: 00 00 00 00 00 00    	m.nop
   25b6a: 25 5a 8c 01  	l.andix r8, r11, 0xc
   25b6e: 01 00        	c.nop
   25b70: 00 00 00 00  	v.nop
   25b74: 00 00 00 00 00 00    	m.nop
   25b7a: 2f 58 00 25  	l.ldi16.s r11, 0x2000
   25b7e: 01 00        	c.nop
   25b80: 00 00 00 00  	v.nop
   25b84: 00 00 00 00 00 00    	m.nop
   25b8a: 6f 5f 91 08  	l.ldi_hi r11, 0x423d
   25b8e: 01 00        	c.nop
   25b90: 00 00 00 00  	v.nop
   25b94: 00 00 00 00 00 00    	m.nop
   25b9a: 2f 58 20 25  	l.ldi16.s r11, 0x2400
   25b9e: 2e 93        	<unknown>
   25ba0: 00 00 00 00  	v.nop
   25ba4: 00 00 00 00 00 00    	m.nop
   25baa: 2f 58 90 00  	l.ldi_hi r11, 0x200
   25bae: e6 03        	<unknown>
   25bb0: 00 00 00 00  	v.nop
   25bb4: 00 00 00 00 00 00    	m.nop
   25bba: b1 22 4c 00  	l.ld.w r10, [r4, 0xc]
   25bbe: 01 00        	c.nop
   25bc0: 00 00 00 00  	v.nop
   25bc4: 00 00 00 00 00 00    	m.nop
   25bca: 65 29 0b 02  	l.orix r5, r5, 0xb
   25bce: 01 00        	c.nop
   25bd0: 00 00 00 00  	v.nop
   25bd4: 00 00 00 00 00 00    	m.nop
   25bda: 2f 5e 33 2b  	l.ldi16.s r11, 0x5678
   25bde: 01 00        	c.nop
   25be0: 00 00 00 00  	v.nop
   25be4: 00 00 00 00 00 00    	m.nop
   25bea: 2f 5d 91 02  	l.ldi_hi r11, 0x1234
   25bee: 01 00        	c.nop
   25bf0: 00 00 00 00  	v.nop
   25bf4: 00 00 00 00 00 00    	m.nop
   25bfa: 31 30 cb 00  	l.st.w r11, [r6]
   25bfe: 01 00        	c.nop
   25c00: 00 00 00 00  	v.nop
   25c04: 00 00 00 00 00 00    	m.nop
   25c0a: e5 3a 0c 02  	l.orix r11, r7, 0xc
   25c0e: 01 00        	c.nop
   25c10: 00 00 00 00  	v.nop
   25c14: 00 00 00 00 00 00    	m.nop
   25c1a: ef 67 7f 1f  	l.ldi16.u r12, 0xffff
   25c1e: 01 00        	c.nop
   25c20: 00 00 00 00  	v.nop
   25c24: 00 00 00 00 00 00    	m.nop
   25c2a: ef 67 ff 00  	l.ldi_hi r12, 0xfff
   25c2e: 01 00        	c.nop
   25c30: 00 00 00 00  	v.nop
   25c34: 00 00 00 00 00 00    	m.nop
   25c3a: e5 41 07 02  	l.orix r7, r8, 0x7
   25c3e: 01 00        	c.nop
   25c40: 00 00 00 00  	v.nop
   25c44: 00 00 00 00 00 00    	m.nop
   25c4a: 31 31 cc 00  	l.st.w r12, [r6, 0x4]
   25c4e: 01 00        	c.nop
   25c50: 00 00 00 00  	v.nop
   25c54: 00 00 00 00 00 00    	m.nop
   25c5a: 31 32 cb 00  	l.st.w r11, [r6, 0x8]
   25c5e: 01 00        	c.nop
   25c60: 00 00 00 00  	v.nop
   25c64: 00 00 00 00 00 00    	m.nop
   25c6a: 31 30 ca 00  	l.st.w r10, [r6]
   25c6e: 01 00        	c.nop
   25c70: 00 00 00 00  	v.nop
   25c74: 00 00 00 00 00 00    	m.nop
   25c7a: 31 31 c5 00  	l.st.w r5, [r6, 0x4]
   25c7e: 01 00        	c.nop
   25c80: 00 00 00 00  	v.nop
   25c84: 00 00 00 00 00 00    	m.nop
   25c8a: 31 32 c7 00  	l.st.w r7, [r6, 0x8]
   25c8e: 01 00        	c.nop
   25c90: 00 00 00 00  	v.nop
   25c94: 00 00 00 00 00 00    	m.nop
   25c9a: 31 21 4c 00  	l.ld.w r4, [r4, 0xc]
   25c9e: 01 00        	c.nop
   25ca0: 00 00 00 00  	v.nop
   25ca4: 00 00 00 00 00 00    	m.nop
   25caa: 2d 20 20 00  	l.wait.dec r4
   25cae: 01 00        	c.nop
   25cb0: 00 00 00 00  	v.nop
   25cb4: 00 00 00 00 00 00    	m.nop
   25cba: 31 11 48 04  	l.ld.w r4, [sp, 0x48]
   25cbe: 01 00        	c.nop
   25cc0: 00 00 00 00  	v.nop
   25cc4: 00 00 00 00 00 00    	m.nop
   25cca: 3b 22 09 40  	<unknown>
   25cce: 01 00        	c.nop
   25cd0: 00 00 00 00  	v.nop
   25cd4: 00 00 00 00 00 00    	m.nop
   25cda: 31 11 5c 0a  	l.ld.w r4, [sp, 0xbc]
   25cde: 01 00        	c.nop
   25ce0: 00 00 00 00  	v.nop
   25ce4: 00 00 00 00 00 00    	m.nop
   25cea: 2f 28 10 11  	l.ldi16.u r5, 0x8200
   25cee: 01 00        	c.nop
   25cf0: 00 00 00 00  	v.nop
   25cf4: 00 00 00 00 00 00    	m.nop
   25cfa: 2f 2d 80 00  	l.ldi_hi r5, 0x14
   25cfe: 01 00        	c.nop
   25d00: 00 00 00 00  	v.nop
   25d04: 00 00 00 00 00 00    	m.nop
   25d0a: e3 24 05 18  	<unknown>
   25d0e: 01 00        	c.nop
   25d10: 00 00 00 00  	v.nop
   25d14: 00 00 00 00 00 00    	m.nop
   25d1a: 31 12 50 0c  	l.ld.w r8, [sp, 0xd0]
   25d1e: 01 00        	c.nop
   25d20: 00 00 00 00  	v.nop
   25d24: 00 00 00 00 00 00    	m.nop
   25d2a: 3b 8a 09 40  	<unknown>
   25d2e: 01 00        	c.nop
   25d30: 00 00 00 00  	v.nop
   25d34: 00 00 00 00 00 00    	m.nop
   25d3a: 2b 40 80 01  	l.jal r0, 0x8 <0x25db0>
   25d3e: 01 00        	c.nop
   25d40: 00 00 00 00  	v.nop
   25d44: 00 00 00 00 00 00    	m.nop
   25d4a: 31 11 58 0a  	l.ld.w r4, [sp, 0xb8]
   25d4e: 01 00        	c.nop
   25d50: 00 00 00 00  	v.nop
   25d54: 00 00 00 00 00 00    	m.nop
   25d5a: 2f 28 10 11  	l.ldi16.u r5, 0x8200
   25d5e: 01 00        	c.nop
   25d60: 00 00 00 00  	v.nop
   25d64: 00 00 00 00 00 00    	m.nop
   25d6a: 6f 2d 80 00  	l.ldi_hi r5, 0x15
   25d6e: 01 00        	c.nop
   25d70: 00 00 00 00  	v.nop
   25d74: 00 00 00 00 00 00    	m.nop
   25d7a: e3 24 05 18  	<unknown>
   25d7e: 01 00        	c.nop
   25d80: 00 00 00 00  	v.nop
   25d84: 00 00 00 00 00 00    	m.nop
   25d8a: 31 12 50 0c  	l.ld.w r8, [sp, 0xd0]
   25d8e: 01 00        	c.nop
   25d90: 00 00 00 00  	v.nop
   25d94: 00 00 00 00 00 00    	m.nop
   25d9a: bb 88 09 00  	<unknown>
   25d9e: 01 00        	c.nop
   25da0: 00 00 00 00  	v.nop
   25da4: 00 00 00 00 00 00    	m.nop
   25daa: 31 12 48 12  	l.ld.w r8, [sp, 0x128]
   25dae: 01 00        	c.nop
   25db0: 00 00 00 00  	v.nop
   25db4: 00 00 00 00 00 00    	m.nop
   25dba: 31 11 44 16  	l.ld.w r4, [sp, 0x164]
   25dbe: 42 83        	<unknown>
   25dc0: 00 00 00 00  	v.nop
   25dc4: 00 00 00 00 00 00    	m.nop
   25dca: 2f 28 70 2b  	l.ldi16.s r5, 0x5e00
   25dce: 22 85        	<unknown>
   25dd0: 00 00 00 00  	v.nop
   25dd4: 00 00 00 00 00 00    	m.nop
   25dda: ef 2c 80 00  	l.ldi_hi r5, 0x13
   25dde: 01 00        	c.nop
   25de0: 00 00 00 00  	v.nop
   25de4: 00 00 00 00 00 00    	m.nop
   25dea: e3 22 05 18  	<unknown>
   25dee: 4e 82        	<unknown>
   25df0: 00 00 00 00  	v.nop
   25df4: 00 00 00 00 00 00    	m.nop
   25dfa: 2f 28 08 21  	l.ldi16.s r5, 0x100
   25dfe: 01 00        	c.nop
   25e00: 00 00 00 00  	v.nop
   25e04: 00 00 00 00 00 00    	m.nop
   25e0a: 00 00        	s.nop16
   25e0c: 00 00        	s.nop16
   25e0e: 96 83        	<unknown>
   25e10: 00 00 00 00  	v.nop
   25e14: 00 00 00 00 00 00    	m.nop
   25e1a: 31 13 c7 00  	l.st.w r7, [sp, 0xc]
   25e1e: a6 82        	<unknown>
   25e20: 00 00 00 00  	v.nop
   25e24: 00 00 00 00 00 00    	m.nop
   25e2a: 31 12 c7 00  	l.st.w r7, [sp, 0x8]
   25e2e: a6 83        	<unknown>
   25e30: 00 00 00 00  	v.nop
   25e34: 00 00 00 00 00 00    	m.nop
   25e3a: 31 11 c9 00  	l.st.w r9, [sp, 0x4]
   25e3e: 01 00        	c.nop
   25e40: 00 00 00 00  	v.nop
   25e44: 00 00 00 00 00 00    	m.nop
   25e4a: 31 10 cb 00  	l.st.w r11, [sp]
   25e4e: a6 85        	<unknown>
   25e50: 00 00 00 00  	v.nop
   25e54: 00 00 00 00 00 00    	m.nop
   25e5a: 31 13 40 22  	l.ld.w r12, [sp, 0x220]
   25e5e: 01 00        	c.nop
   25e60: 00 00 00 00  	v.nop
   25e64: 00 00 00 00 00 00    	m.nop
   25e6a: eb 60 00 01  	l.jalr lr, r12, 0x0
   25e6e: 01 00        	c.nop
   25e70: 00 00 00 00  	v.nop
   25e74: 00 00 00 00 00 00    	m.nop
   25e7a: 3b f3 00 33  	<unknown>
   25e7e: 01 00        	c.nop
   25e80: 00 00 00 00  	v.nop
   25e84: 00 00 00 00 00 00    	m.nop
   25e8a: 31 11 48 04  	l.ld.w r4, [sp, 0x48]
   25e8e: 81 44        	<unknown>
   25e90: 00 00 00 00  	v.nop
   25e94: 00 00 00 00 00 00    	m.nop
   25e9a: bb 27 69 32  	<unknown>
   25e9e: 01 00        	c.nop
   25ea0: 00 00 00 00  	v.nop
   25ea4: 00 00 00 00 00 00    	m.nop
   25eaa: 31 11 5c 0a  	l.ld.w r4, [sp, 0xbc]
   25eae: 01 00        	c.nop
   25eb0: 00 00 00 00  	v.nop
   25eb4: 00 00 00 00 00 00    	m.nop
   25eba: 2f 28 10 11  	l.ldi16.u r5, 0x8200
   25ebe: 01 00        	c.nop
   25ec0: 00 00 00 00  	v.nop
   25ec4: 00 00 00 00 00 00    	m.nop
   25eca: 2f 2d 80 00  	l.ldi_hi r5, 0x14
   25ece: 01 00        	c.nop
   25ed0: 00 00 00 00  	v.nop
   25ed4: 00 00 00 00 00 00    	m.nop
   25eda: 2b 68 97 3f  	l.jal r0, -0x683 <0x1f6a0>
   25ede: 01 00        	c.nop
   25ee0: 00 00 00 00  	v.nop
   25ee4: 00 00 00 00 00 00    	m.nop
   25eea: 00 00        	s.nop16
   25eec: 00 00        	s.nop16
   25eee: 85 44        	<unknown>
   25ef0: 00 00 00 00  	v.nop
   25ef4: 00 00 00 00 00 00    	m.nop
   25efa: b1 32 5c 02  	l.ld.w r10, [r6, 0x3c]
   25efe: 01 00        	c.nop
   25f00: 00 00 00 00  	v.nop
   25f04: 00 00 00 00 00 00    	m.nop
   25f0a: f1 32 50 02  	l.ld.w r11, [r6, 0x30]
   25f0e: 01 00        	c.nop
   25f10: 00 00 00 00  	v.nop
   25f14: 00 00 00 00 00 00    	m.nop
   25f1a: 31 33 54 02  	l.ld.w r12, [r6, 0x34]
   25f1e: 01 00        	c.nop
   25f20: 00 00 00 00  	v.nop
   25f24: 00 00 00 00 00 00    	m.nop
   25f2a: 2f 80 00 25  	l.ldi16.s r16, 0x2000
   25f2e: 01 00        	c.nop
   25f30: 00 00 00 00  	v.nop
   25f34: 00 00 00 00 00 00    	m.nop
   25f3a: 71 33 5c 00  	l.ld.w r13, [r6, 0x1c]
   25f3e: 01 00        	c.nop
   25f40: 00 00 00 00  	v.nop
   25f44: 00 00 00 00 00 00    	m.nop
   25f4a: 63 52 09 18  	<unknown>
   25f4e: 01 00        	c.nop
   25f50: 00 00 00 00  	v.nop
   25f54: 00 00 00 00 00 00    	m.nop
   25f5a: a3 5a 19 03  	l.slli r10, r11, 0x19
   25f5e: 01 00        	c.nop
   25f60: 00 00 00 00  	v.nop
   25f64: 00 00 00 00 00 00    	m.nop
   25f6a: e3 62 04 03  	l.slli r11, r12, 0x4
   25f6e: 01 00        	c.nop
   25f70: 00 00 00 00  	v.nop
   25f74: 00 00 00 00 00 00    	m.nop
   25f7a: ef 67 7f 1f  	l.ldi16.u r12, 0xffff
   25f7e: 01 00        	c.nop
   25f80: 00 00 00 00  	v.nop
   25f84: 00 00 00 00 00 00    	m.nop
   25f8a: ef 67 83 00  	l.ldi_hi r12, 0x7f
   25f8e: 01 00        	c.nop
   25f90: 00 00 00 00  	v.nop
   25f94: 00 00 00 00 00 00    	m.nop
   25f9a: 6f 87 91 08  	l.ldi_hi r16, 0x423d
   25f9e: 32 87        	<unknown>
   25fa0: 00 00 00 00  	v.nop
   25fa4: 00 00 00 00 00 00    	m.nop
   25faa: 25 4b 8e 01  	l.andix r12, r9, 0xe
   25fae: c2 95        	<unknown>
   25fb0: 00 00 00 00  	v.nop
   25fb4: 00 00 00 00 00 00    	m.nop
   25fba: 31 37 c9 02  	l.st.w r9, [r6, 0x3c]
   25fbe: 01 00        	c.nop
   25fc0: 00 00 00 00  	v.nop
   25fc4: 00 00 00 00 00 00    	m.nop
   25fca: 2f 4e 33 2b  	l.ldi16.s r9, 0x5678
   25fce: 01 00        	c.nop
   25fd0: 00 00 00 00  	v.nop
   25fd4: 00 00 00 00 00 00    	m.nop
   25fda: 2f 4d 91 02  	l.ldi_hi r9, 0x1234
   25fde: 01 00        	c.nop
   25fe0: 00 00 00 00  	v.nop
   25fe4: 00 00 00 00 00 00    	m.nop
   25fea: 31 58 c9 00  	l.st.w r9, [r11]
   25fee: 01 00        	c.nop
   25ff0: 00 00 00 00  	v.nop
   25ff4: 00 00 00 00 00 00    	m.nop
   25ffa: 65 52 0e 02  	l.orix r9, r10, 0xe
   25ffe: 01 00        	c.nop
   26000: 00 00 00 00  	v.nop
   26004: 00 00 00 00 00 00    	m.nop
   2600a: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   2600e: 01 00        	c.nop
   26010: 00 00 00 00  	v.nop
   26014: 00 00 00 00 00 00    	m.nop
   2601a: ef 77 ff 00  	l.ldi_hi r14, 0xfff
   2601e: 01 00        	c.nop
   26020: 00 00 00 00  	v.nop
   26024: 00 00 00 00 00 00    	m.nop
   2602a: 31 59 ce 00  	l.st.w r14, [r11, 0x4]
   2602e: 01 00        	c.nop
   26030: 00 00 00 00  	v.nop
   26034: 00 00 00 00 00 00    	m.nop
   2603a: 31 5a c9 00  	l.st.w r9, [r11, 0x8]
   2603e: 01 00        	c.nop
   26040: 00 00 00 00  	v.nop
   26044: 00 00 00 00 00 00    	m.nop
   2604a: 65 52 0c 02  	l.orix r9, r10, 0xc
   2604e: 01 00        	c.nop
   26050: 00 00 00 00  	v.nop
   26054: 00 00 00 00 00 00    	m.nop
   2605a: 31 58 cd 00  	l.st.w r13, [r11]
   2605e: 01 00        	c.nop
   26060: 00 00 00 00  	v.nop
   26064: 00 00 00 00 00 00    	m.nop
   2606a: 31 59 c5 00  	l.st.w r5, [r11, 0x4]
   2606e: 01 00        	c.nop
   26070: 00 00 00 00  	v.nop
   26074: 00 00 00 00 00 00    	m.nop
   2607a: 31 5a c9 00  	l.st.w r9, [r11, 0x8]
   2607e: 01 00        	c.nop
   26080: 00 00 00 00  	v.nop
   26084: 00 00 00 00 00 00    	m.nop
   2608a: b1 32 0c 06  	l.ld.b r10, [r6, 0x6c]
   2608e: 01 00        	c.nop
   26090: 00 00 00 00  	v.nop
   26094: 00 00 00 00 00 00    	m.nop
   2609a: 61 3a 98 00  	l.addi.u r9, r7, 0x98
   2609e: 01 00        	c.nop
   260a0: 00 00 00 00  	v.nop
   260a4: 00 00 00 00 00 00    	m.nop
   260aa: a5 52 9c 01  	l.andix r10, r10, 0x1c
   260ae: 01 00        	c.nop
   260b0: 00 00 00 00  	v.nop
   260b4: 00 00 00 00 00 00    	m.nop
   260ba: fb 56 60 78  	<unknown>
   260be: 01 00        	c.nop
   260c0: 00 00 00 00  	v.nop
   260c4: 00 00 00 00 00 00    	m.nop
   260ca: 2b 30 ca 3f  	l.jal r0, -0x35a <0x22b20>
   260ce: 01 00        	c.nop
   260d0: 00 00 00 00  	v.nop
   260d4: 00 00 00 00 00 00    	m.nop
   260da: 00 00        	s.nop16
   260dc: 00 00        	s.nop16
   260de: 85 44        	<unknown>
   260e0: 00 00 00 00  	v.nop
   260e4: 00 00 00 00 00 00    	m.nop
   260ea: b1 3a 5c 02  	l.ld.w r10, [r7, 0x3c]
   260ee: 01 00        	c.nop
   260f0: 00 00 00 00  	v.nop
   260f4: 00 00 00 00 00 00    	m.nop
   260fa: f1 3a 50 02  	l.ld.w r11, [r7, 0x30]
   260fe: 01 00        	c.nop
   26100: 00 00 00 00  	v.nop
   26104: 00 00 00 00 00 00    	m.nop
   2610a: 31 3b 54 02  	l.ld.w r12, [r7, 0x34]
   2610e: 01 00        	c.nop
   26110: 00 00 00 00  	v.nop
   26114: 00 00 00 00 00 00    	m.nop
   2611a: 2f 80 00 25  	l.ldi16.s r16, 0x2000
   2611e: 01 00        	c.nop
   26120: 00 00 00 00  	v.nop
   26124: 00 00 00 00 00 00    	m.nop
   2612a: 71 3b 5c 00  	l.ld.w r13, [r7, 0x1c]
   2612e: 01 00        	c.nop
   26130: 00 00 00 00  	v.nop
   26134: 00 00 00 00 00 00    	m.nop
   2613a: 63 52 09 18  	<unknown>
   2613e: 01 00        	c.nop
   26140: 00 00 00 00  	v.nop
   26144: 00 00 00 00 00 00    	m.nop
   2614a: a3 5a 19 03  	l.slli r10, r11, 0x19
   2614e: 01 00        	c.nop
   26150: 00 00 00 00  	v.nop
   26154: 00 00 00 00 00 00    	m.nop
   2615a: e3 62 04 03  	l.slli r11, r12, 0x4
   2615e: 01 00        	c.nop
   26160: 00 00 00 00  	v.nop
   26164: 00 00 00 00 00 00    	m.nop
   2616a: ef 67 7f 1f  	l.ldi16.u r12, 0xffff
   2616e: 01 00        	c.nop
   26170: 00 00 00 00  	v.nop
   26174: 00 00 00 00 00 00    	m.nop
   2617a: ef 67 83 00  	l.ldi_hi r12, 0x7f
   2617e: 01 00        	c.nop
   26180: 00 00 00 00  	v.nop
   26184: 00 00 00 00 00 00    	m.nop
   2618a: 6f 87 91 08  	l.ldi_hi r16, 0x423d
   2618e: 32 87        	<unknown>
   26190: 00 00 00 00  	v.nop
   26194: 00 00 00 00 00 00    	m.nop
   2619a: 25 4b 8e 01  	l.andix r12, r9, 0xe
   2619e: c2 95        	<unknown>
   261a0: 00 00 00 00  	v.nop
   261a4: 00 00 00 00 00 00    	m.nop
   261aa: 31 3f c9 02  	l.st.w r9, [r7, 0x3c]
   261ae: 01 00        	c.nop
   261b0: 00 00 00 00  	v.nop
   261b4: 00 00 00 00 00 00    	m.nop
   261ba: 2f 4e 33 2b  	l.ldi16.s r9, 0x5678
   261be: 01 00        	c.nop
   261c0: 00 00 00 00  	v.nop
   261c4: 00 00 00 00 00 00    	m.nop
   261ca: 2f 4d 91 02  	l.ldi_hi r9, 0x1234
   261ce: 01 00        	c.nop
   261d0: 00 00 00 00  	v.nop
   261d4: 00 00 00 00 00 00    	m.nop
   261da: 31 58 c9 00  	l.st.w r9, [r11]
   261de: 01 00        	c.nop
   261e0: 00 00 00 00  	v.nop
   261e4: 00 00 00 00 00 00    	m.nop
   261ea: 65 52 0e 02  	l.orix r9, r10, 0xe
   261ee: 01 00        	c.nop
   261f0: 00 00 00 00  	v.nop
   261f4: 00 00 00 00 00 00    	m.nop
   261fa: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   261fe: 01 00        	c.nop
   26200: 00 00 00 00  	v.nop
   26204: 00 00 00 00 00 00    	m.nop
   2620a: ef 77 ff 00  	l.ldi_hi r14, 0xfff
   2620e: 01 00        	c.nop
   26210: 00 00 00 00  	v.nop
   26214: 00 00 00 00 00 00    	m.nop
   2621a: 31 59 ce 00  	l.st.w r14, [r11, 0x4]
   2621e: 01 00        	c.nop
   26220: 00 00 00 00  	v.nop
   26224: 00 00 00 00 00 00    	m.nop
   2622a: 31 5a c9 00  	l.st.w r9, [r11, 0x8]
   2622e: 01 00        	c.nop
   26230: 00 00 00 00  	v.nop
   26234: 00 00 00 00 00 00    	m.nop
   2623a: 65 52 0c 02  	l.orix r9, r10, 0xc
   2623e: 01 00        	c.nop
   26240: 00 00 00 00  	v.nop
   26244: 00 00 00 00 00 00    	m.nop
   2624a: 31 58 cd 00  	l.st.w r13, [r11]
   2624e: 01 00        	c.nop
   26250: 00 00 00 00  	v.nop
   26254: 00 00 00 00 00 00    	m.nop
   2625a: 31 59 c6 00  	l.st.w r6, [r11, 0x4]
   2625e: 01 00        	c.nop
   26260: 00 00 00 00  	v.nop
   26264: 00 00 00 00 00 00    	m.nop
   2626a: 31 5a c9 00  	l.st.w r9, [r11, 0x8]
   2626e: 01 00        	c.nop
   26270: 00 00 00 00  	v.nop
   26274: 00 00 00 00 00 00    	m.nop
   2627a: 71 3a 0c 06  	l.ld.b r9, [r7, 0x6c]
   2627e: 01 00        	c.nop
   26280: 00 00 00 00  	v.nop
   26284: 00 00 00 00 00 00    	m.nop
   2628a: 65 4a 9c 01  	l.andix r9, r9, 0x1c
   2628e: 01 00        	c.nop
   26290: 00 00 00 00  	v.nop
   26294: 00 00 00 00 00 00    	m.nop
   2629a: 3b 4f 60 79  	<unknown>
   2629e: 01 00        	c.nop
   262a0: 00 00 00 00  	v.nop
   262a4: 00 00 00 00 00 00    	m.nop
   262aa: 2b 38 d2 3f  	l.jal r0, -0x2d9 <0x23510>
   262ae: 01 00        	c.nop
   262b0: 00 00 00 00  	v.nop
   262b4: 00 00 00 00 00 00    	m.nop
   262ba: 31 11 48 04  	l.ld.w r4, [sp, 0x48]
   262be: 01 00        	c.nop
   262c0: 00 00 00 00  	v.nop
   262c4: 00 00 00 00 00 00    	m.nop
   262ca: 7b 23 0f 00  	<unknown>
   262ce: 01 00        	c.nop
   262d0: 00 00 00 00  	v.nop
   262d4: 00 00 00 00 00 00    	m.nop
   262da: 31 11 5c 0a  	l.ld.w r4, [sp, 0xbc]
   262de: 01 00        	c.nop
   262e0: 00 00 00 00  	v.nop
   262e4: 00 00 00 00 00 00    	m.nop
   262ea: f1 14 48 06  	l.ld.w r19, [sp, 0x68]
   262ee: 01 00        	c.nop
   262f0: 00 00 00 00  	v.nop
   262f4: 00 00 00 00 00 00    	m.nop
   262fa: 31 15 5c 06  	l.ld.w r20, [sp, 0x7c]
   262fe: 01 00        	c.nop
   26300: 00 00 00 00  	v.nop
   26304: 00 00 00 00 00 00    	m.nop
   2630a: b1 17 54 0e  	l.ld.w r30, [sp, 0xf4]
   2630e: 01 00        	c.nop
   26310: 00 00 00 00  	v.nop
   26314: 00 00 00 00 00 00    	m.nop
   2631a: 2f 28 10 11  	l.ldi16.u r5, 0x8200
   2631e: 01 00        	c.nop
   26320: 00 00 00 00  	v.nop
   26324: 00 00 00 00 00 00    	m.nop
   2632a: ef bf 7f 1f  	l.ldi16.u r23, 0xffff
   2632e: 01 00        	c.nop
   26330: 00 00 00 00  	v.nop
   26334: 00 00 00 00 00 00    	m.nop
   2633a: 2f cd 08 21  	l.ldi16.s r25, 0x114
   2633e: 01 00        	c.nop
   26340: 00 00 00 00  	v.nop
   26344: 00 00 00 00 00 00    	m.nop
   2634a: 2f 2d 80 00  	l.ldi_hi r5, 0x14
   2634e: 01 00        	c.nop
   26350: 00 00 00 00  	v.nop
   26354: 00 00 00 00 00 00    	m.nop
   2635a: e3 21 05 18  	<unknown>
   2635e: 01 00        	c.nop
   26360: 00 00 00 00  	v.nop
   26364: 00 00 00 00 00 00    	m.nop
   2636a: ef bf 87 00  	l.ldi_hi r23, 0xff
   2636e: 01 00        	c.nop
   26370: 00 00 00 00  	v.nop
   26374: 00 00 00 00 00 00    	m.nop
   2637a: 2f c8 81 10  	l.ldi_hi r25, 0x8020
   2637e: 01 00        	c.nop
   26380: 00 00 00 00  	v.nop
   26384: 00 00 00 00 00 00    	m.nop
   2638a: 2b 60 80 01  	l.jal r0, 0xc <0x26440>
   2638e: 01 00        	c.nop
   26390: 00 00 00 00  	v.nop
   26394: 00 00 00 00 00 00    	m.nop
   2639a: 31 11 58 0a  	l.ld.w r4, [sp, 0xb8]
   2639e: 01 00        	c.nop
   263a0: 00 00 00 00  	v.nop
   263a4: 00 00 00 00 00 00    	m.nop
   263aa: 2f 28 10 11  	l.ldi16.u r5, 0x8200
   263ae: 01 00        	c.nop
   263b0: 00 00 00 00  	v.nop
   263b4: 00 00 00 00 00 00    	m.nop
   263ba: ef bf 7f 1f  	l.ldi16.u r23, 0xffff
   263be: 01 00        	c.nop
   263c0: 00 00 00 00  	v.nop
   263c4: 00 00 00 00 00 00    	m.nop
   263ca: 2f cd 08 21  	l.ldi16.s r25, 0x114
   263ce: 01 00        	c.nop
   263d0: 00 00 00 00  	v.nop
   263d4: 00 00 00 00 00 00    	m.nop
   263da: 6f 2d 80 00  	l.ldi_hi r5, 0x15
   263de: 01 00        	c.nop
   263e0: 00 00 00 00  	v.nop
   263e4: 00 00 00 00 00 00    	m.nop
   263ea: e3 21 05 18  	<unknown>
   263ee: 01 00        	c.nop
   263f0: 00 00 00 00  	v.nop
   263f4: 00 00 00 00 00 00    	m.nop
   263fa: f1 14 48 06  	l.ld.w r19, [sp, 0x68]
   263fe: 01 00        	c.nop
   26400: 00 00 00 00  	v.nop
   26404: 00 00 00 00 00 00    	m.nop
   2640a: 31 15 5c 06  	l.ld.w r20, [sp, 0x7c]
   2640e: 01 00        	c.nop
   26410: 00 00 00 00  	v.nop
   26414: 00 00 00 00 00 00    	m.nop
   2641a: ef bf 87 00  	l.ldi_hi r23, 0xff
   2641e: 01 00        	c.nop
   26420: 00 00 00 00  	v.nop
   26424: 00 00 00 00 00 00    	m.nop
   2642a: 2f c8 81 10  	l.ldi_hi r25, 0x8020
   2642e: 01 00        	c.nop
   26430: 00 00 00 00  	v.nop
   26434: 00 00 00 00 00 00    	m.nop
   2643a: b1 17 54 0e  	l.ld.w r30, [sp, 0xf4]
   2643e: 01 00        	c.nop
   26440: 00 00 00 00  	v.nop
   26444: 00 00 00 00 00 00    	m.nop
   2644a: 31 11 50 14  	l.ld.w r4, [sp, 0x150]
   2644e: a3 87        	<unknown>
   26450: 00 00 00 00  	v.nop
   26454: 00 00 00 00 00 00    	m.nop
   2645a: 2f 28 78 2f  	l.ldi16.s r5, 0x7f00
   2645e: 97 87        	<unknown>
   26460: 00 00 00 00  	v.nop
   26464: 00 00 00 00 00 00    	m.nop
   2646a: 2f 2d 80 00  	l.ldi_hi r5, 0x14
   2646e: 9b 87        	<unknown>
   26470: 00 00 00 00  	v.nop
   26474: 00 00 00 00 00 00    	m.nop
   2647a: 23 3a 88 0b  	l.srai r8, r7, 0x8
   2647e: 9f 87        	<unknown>
   26480: 00 00 00 00  	v.nop
   26484: 00 00 00 00 00 00    	m.nop
   2648a: 63 3a 87 0b  	l.srai r9, r7, 0x7
   2648e: 83 8f        	<unknown>
   26490: 00 00 00 00  	v.nop
   26494: 00 00 00 00 00 00    	m.nop
   2649a: 00 00        	s.nop16
   2649c: 00 00        	s.nop16
   2649e: 16 92        	<unknown>
   264a0: 00 00 00 00  	v.nop
   264a4: 00 00 00 00 00 00    	m.nop
   264aa: 63 21 86 0b  	l.srai r5, r4, 0x6
   264ae: 01 00        	c.nop
   264b0: 00 00 00 00  	v.nop
   264b4: 00 00 00 00 00 00    	m.nop
   264ba: a1 29 01 00  	l.addi.u r6, r5, 0x1
   264be: 01 00        	c.nop
   264c0: 00 00 00 00  	v.nop
   264c4: 00 00 00 00 00 00    	m.nop
   264ca: 33 40 05 04  	l.movsr2tar.ext.32sext.vpt0 ta8, r5
   264ce: 01 00        	c.nop
   264d0: 00 00 00 00  	v.nop
   264d4: 00 00 00 00 00 00    	m.nop
   264da: 33 40 06 0c  	l.movsr2tar.ext.32sext.vpt1 ta8, r6
   264de: 01 00        	c.nop
   264e0: 00 00 00 00  	v.nop
   264e4: 00 00 00 00 00 00    	m.nop
   264ea: 33 48 16 04  	l.movsr2tar.ext.32sext.vpt0 ta9, r22
   264ee: 01 00        	c.nop
   264f0: 00 00 00 00  	v.nop
   264f4: 00 00 00 00 00 00    	m.nop
   264fa: 33 48 16 0c  	l.movsr2tar.ext.32sext.vpt1 ta9, r22
   264fe: 01 00        	c.nop
   26500: 00 00 00 00  	v.nop
   26504: 00 00 00 00 00 00    	m.nop
   2650a: 61 41 40 00  	l.addi.u r5, r8, 0x40
   2650e: 11 43        	<unknown>
   26510: 00 00 00 00  	v.nop
   26514: 00 00 00 00 00 00    	m.nop
   2651a: 6f 51 7e 3f  	l.ldi16.s r10, -0x3b
   2651e: 01 00        	c.nop
   26520: 00 00 00 00  	v.nop
   26524: 00 00 00 00 00 00    	m.nop
   2652a: 2f 5f 03 21  	l.ldi16.s r11, 0x7c
   2652e: 01 00        	c.nop
   26530: 00 00 00 00  	v.nop
   26534: 00 00 00 00 00 00    	m.nop
   2653a: 33 80 08 04  	l.movsr2tar.ext.32sext.vpt0 ta16, r8
   2653e: 01 00        	c.nop
   26540: 00 00 00 00  	v.nop
   26544: 00 00 00 00 00 00    	m.nop
   2654a: 33 80 05 0c  	l.movsr2tar.ext.32sext.vpt1 ta16, r5
   2654e: 01 00        	c.nop
   26550: 00 00 00 00  	v.nop
   26554: 00 00 00 00 00 00    	m.nop
   2655a: 33 88 06 04  	l.movsr2tar.ext.32sext.vpt0 ta17, r6
   2655e: 01 00        	c.nop
   26560: 00 00 00 00  	v.nop
   26564: 00 00 00 00 00 00    	m.nop
   2656a: 33 88 06 0c  	l.movsr2tar.ext.32sext.vpt1 ta17, r6
   2656e: 01 00        	c.nop
   26570: 00 00 00 00  	v.nop
   26574: 00 00 00 00 00 00    	m.nop
   2657a: 33 90 0a 04  	l.movsr2tar.ext.32sext.vpt0 ta18, r10
   2657e: 01 00        	c.nop
   26580: 00 00 00 00  	v.nop
   26584: 00 00 00 00 00 00    	m.nop
   2658a: 33 90 0a 0c  	l.movsr2tar.ext.32sext.vpt1 ta18, r10
   2658e: 01 00        	c.nop
   26590: 00 00 00 00  	v.nop
   26594: 00 00 00 00 00 00    	m.nop
   2659a: 33 98 0b 04  	l.movsr2tar.ext.32sext.vpt0 ta19, r11
   2659e: 01 00        	c.nop
   265a0: 00 00 00 00  	v.nop
   265a4: 00 00 00 00 00 00    	m.nop
   265aa: 33 98 0b 0c  	l.movsr2tar.ext.32sext.vpt1 ta19, r11
   265ae: 01 00        	c.nop
   265b0: 00 00 00 00  	v.nop
   265b4: 00 00 00 00 00 00    	m.nop
   265ba: 39 01 09 00  	<unknown>
   265be: 01 00        	c.nop
   265c0: 00 00 00 00  	v.nop
   265c4: 00 00 00 00 00 00    	m.nop
   265ca: 61 49 80 00  	l.addi.u r5, r9, 0x80
   265ce: 21 43        	<unknown>
   265d0: 00 00 00 00  	v.nop
   265d4: 00 00 00 00 00 00    	m.nop
   265da: 6f 42 7c 3f  	l.ldi16.s r8, -0x77
   265de: 01 00        	c.nop
   265e0: 00 00 00 00  	v.nop
   265e4: 00 00 00 00 00 00    	m.nop
   265ea: 6f 51 04 21  	l.ldi16.s r10, 0x85
   265ee: 01 00        	c.nop
   265f0: 00 00 00 00  	v.nop
   265f4: 00 00 00 00 00 00    	m.nop
   265fa: 33 c0 09 04  	l.movsr2tar.ext.32sext.vpt0 ta24, r9
   265fe: 01 00        	c.nop
   26600: 00 00 00 00  	v.nop
   26604: 00 00 00 00 00 00    	m.nop
   2660a: 33 c0 05 0c  	l.movsr2tar.ext.32sext.vpt1 ta24, r5
   2660e: 01 00        	c.nop
   26610: 00 00 00 00  	v.nop
   26614: 00 00 00 00 00 00    	m.nop
   2661a: 33 c8 06 04  	l.movsr2tar.ext.32sext.vpt0 ta25, r6
   2661e: 01 00        	c.nop
   26620: 00 00 00 00  	v.nop
   26624: 00 00 00 00 00 00    	m.nop
   2662a: 33 c8 06 0c  	l.movsr2tar.ext.32sext.vpt1 ta25, r6
   2662e: 01 00        	c.nop
   26630: 00 00 00 00  	v.nop
   26634: 00 00 00 00 00 00    	m.nop
   2663a: 33 d0 08 04  	l.movsr2tar.ext.32sext.vpt0 ta26, r8
   2663e: 01 00        	c.nop
   26640: 00 00 00 00  	v.nop
   26644: 00 00 00 00 00 00    	m.nop
   2664a: 33 d0 08 0c  	l.movsr2tar.ext.32sext.vpt1 ta26, r8
   2664e: 01 00        	c.nop
   26650: 00 00 00 00  	v.nop
   26654: 00 00 00 00 00 00    	m.nop
   2665a: 33 d8 0a 04  	l.movsr2tar.ext.32sext.vpt0 ta27, r10
   2665e: 01 00        	c.nop
   26660: 00 00 00 00  	v.nop
   26664: 00 00 00 00 00 00    	m.nop
   2666a: 33 d8 0a 0c  	l.movsr2tar.ext.32sext.vpt1 ta27, r10
   2666e: 01 00        	c.nop
   26670: 00 00 00 00  	v.nop
   26674: 00 00 00 00 00 00    	m.nop
   2667a: 00 00        	s.nop16
   2667c: 00 00        	s.nop16
   2667e: da 82        	<unknown>
   26680: 00 00 00 00  	v.nop
   26684: 00 00 00 00 00 00    	m.nop
   2668a: 00 00        	s.nop16
   2668c: 00 00        	s.nop16
   2668e: 0d 43        	<unknown>
   26690: 00 00 00 00  	v.nop
   26694: 10 01 06 10 80 6a    	m.msf.rec oacc4, oacc4
   2669a: 39 10 91 93  	<unknown>
   2669e: 01 00        	c.nop
   266a0: 00 00 00 00  	v.nop
   266a4: 1c 11 11 00 20 18    	m.nop
   266aa: 00 00        	s.nop16
   266ac: 00 00        	s.nop16
   266ae: 01 00        	c.nop
   266b0: 00 00 00 00  	v.nop
   266b4: 1c 12 11 00 20 18    	m.nop
   266ba: 00 00        	s.nop16
   266bc: 00 00        	s.nop16
   266be: 01 00        	c.nop
   266c0: 00 00 00 00  	v.nop
   266c4: 1c 13 11 00 20 18    	m.nop
   266ca: a1 32 ff 2f  	l.addi.s r10, r6, -0x1
   266ce: 01 00        	c.nop
   266d0: 00 00 00 00  	v.nop
   266d4: 1c 14 11 00 20 18    	m.nop
   266da: 00 00        	s.nop16
   266dc: 00 00        	s.nop16
   266de: 01 00        	c.nop
   266e0: 00 00 00 00  	v.nop
   266e4: 1c 15 11 00 20 18    	m.nop
   266ea: 00 00        	s.nop16
   266ec: 00 00        	s.nop16
   266ee: 01 00        	c.nop
   266f0: 00 00 00 00  	v.nop
   266f4: 1c 16 11 00 20 18    	m.nop
   266fa: 00 00        	s.nop16
   266fc: 00 00        	s.nop16
   266fe: 01 00        	c.nop
   26700: 00 00 00 00  	v.nop
   26704: 1c 17 11 00 20 18    	m.nop
   2670a: 00 00        	s.nop16
   2670c: 00 00        	s.nop16
   2670e: 01 00        	c.nop
   26710: 00 00 00 00  	v.nop
   26714: 1c 18 11 00 20 18    	m.nop
   2671a: 63 52 9f 03  	l.srli r9, r10, 0x1f
   2671e: 01 00        	c.nop
   26720: 00 00 00 00  	v.nop
   26724: 1c 19 11 00 20 18    	m.nop
   2672a: 6f 40 00 21  	l.ldi16.s r8, 0x1
   2672e: 01 00        	c.nop
   26730: 00 00 00 00  	v.nop
   26734: 1c 1a 11 00 20 18    	m.nop
   2673a: 00 00        	s.nop16
   2673c: 00 00        	s.nop16
   2673e: 01 00        	c.nop
   26740: 00 00 00 00  	v.nop
   26744: 1c 1b 11 00 20 18    	m.nop
   2674a: 35 01 80 11  	<unknown>
   2674e: 01 00        	c.nop
   26750: 00 00 00 00  	v.nop
   26754: 1c 1c 11 00 20 18    	m.nop
   2675a: 00 00        	s.nop16
   2675c: 00 00        	s.nop16
   2675e: 01 00        	c.nop
   26760: 00 00 00 00  	v.nop
   26764: 1c 1d 11 00 20 18    	m.nop
   2676a: 00 00        	s.nop16
   2676c: 00 00        	s.nop16
   2676e: 01 00        	c.nop
   26770: 00 00 00 00  	v.nop
   26774: 1c 1e 11 00 20 18    	m.nop
   2677a: 00 00        	s.nop16
   2677c: 00 00        	s.nop16
   2677e: 3b 84        	<unknown>
   26780: 00 00 00 00  	v.nop
   26784: 1c 1f 12 00 20 18    	m.nop
   2678a: 00 00        	s.nop16
   2678c: 00 00        	s.nop16
   2678e: c3 84        	<unknown>
   26790: 32 32 99 88  	<unknown>
   26794: 4c 08 00 40 a0 22    	m.nop
   2679a: 00 00        	s.nop16
   2679c: 00 00        	s.nop16
   2679e: 01 00        	c.nop
   267a0: 32 33 99 88  	<unknown>
   267a4: 8c 08 40 44 a0 23    	m.nop
   267aa: 39 10 91 93  	<unknown>
   267ae: 01 00        	c.nop
   267b0: 32 34 99 88  	<unknown>
   267b4: cc 08 80 48 a0 23    	m.nop
   267ba: 39 11 91 93  	<unknown>
   267be: 01 00        	c.nop
   267c0: 32 35 99 88  	<unknown>
   267c4: 0c 09 c0 4c a0 23    	m.nop
   267ca: 39 12 91 93  	<unknown>
   267ce: 01 00        	c.nop
   267d0: 32 36 99 88  	<unknown>
   267d4: 4c 09 00 51 a0 23    	m.nop
   267da: 39 13 91 93  	<unknown>
   267de: 01 00        	c.nop
   267e0: 32 37 99 88  	<unknown>
   267e4: 8c 09 40 55 a0 23    	m.nop
   267ea: 39 14 91 93  	<unknown>
   267ee: 01 00        	c.nop
   267f0: 32 38 99 88  	<unknown>
   267f4: cc 09 80 59 a0 23    	m.nop
   267fa: 39 15 91 93  	<unknown>
   267fe: 01 00        	c.nop
   26800: 32 39 99 88  	<unknown>
   26804: 0c 0a c0 5d a0 23    	m.nop
   2680a: 39 16 91 93  	<unknown>
   2680e: 01 00        	c.nop
   26810: 32 3a 99 88  	<unknown>
   26814: 4c 0a 00 62 a0 23    	m.nop
   2681a: 39 17 91 93  	<unknown>
   2681e: 01 00        	c.nop
   26820: 32 3b 99 88  	<unknown>
   26824: 8c 0a 40 66 a0 23    	m.nop
   2682a: 39 18 91 93  	<unknown>
   2682e: 01 00        	c.nop
   26830: 32 3c 99 88  	<unknown>
   26834: cc 0a 80 6a a0 23    	m.nop
   2683a: 39 19 91 93  	<unknown>
   2683e: 01 00        	c.nop
   26840: 32 3d 99 88  	<unknown>
   26844: 0c 0b c0 6e a0 23    	m.nop
   2684a: 39 1a 91 93  	<unknown>
   2684e: 01 00        	c.nop
   26850: 32 3e 99 88  	<unknown>
   26854: 4c 0b 00 73 a0 23    	m.nop
   2685a: 39 1b 91 93  	<unknown>
   2685e: 01 00        	c.nop
   26860: 32 3f 99 88  	<unknown>
   26864: 8c 0b 40 77 a0 23    	m.nop
   2686a: 39 1c 91 93  	<unknown>
   2686e: 01 00        	c.nop
   26870: 32 40 99 88  	<unknown>
   26874: cc 0b 80 7b a0 23    	m.nop
   2687a: 39 1d 91 93  	<unknown>
   2687e: 01 00        	c.nop
   26880: 32 41 9a 88  	<unknown>
   26884: 0c 0c c0 7f a0 23    	m.nop
   2688a: 39 1e 91 93  	<unknown>
   2688e: 01 00        	c.nop
   26890: 00 00 00 00  	v.nop
   26894: 0c 19 23 84 20 73    	m.nop
   2689a: 39 1f 92 93  	<unknown>
   2689e: 01 00        	c.nop
   268a0: 00 00 00 00  	v.nop
   268a4: 8c 19 23 88 20 72    	m.nop
   268aa: e1 52 01 20  	l.addi.s r11, r10, 0x1
   268ae: 01 00        	c.nop
   268b0: 00 00 00 00  	v.nop
   268b4: 0c 1a 23 8c 20 72    	m.nop
   268ba: 2f 40 00 21  	l.ldi16.s r8, 0x0
   268be: 01 00        	c.nop
   268c0: 00 00 00 00  	v.nop
   268c4: 8c 1a 23 90 20 72    	m.nop
   268ca: a1 52 ff 2f  	l.addi.s r10, r10, -0x1
   268ce: 01 00        	c.nop
   268d0: 00 00 00 00  	v.nop
   268d4: 0c 1b 23 94 20 72    	m.nop
   268da: 00 00        	s.nop16
   268dc: 00 00        	s.nop16
   268de: 01 00        	c.nop
   268e0: 00 00 00 00  	v.nop
   268e4: 8c 1b 23 98 20 72    	m.nop
   268ea: 00 00        	s.nop16
   268ec: 00 00        	s.nop16
   268ee: 01 00        	c.nop
   268f0: 00 00 00 00  	v.nop
   268f4: 0c 1c 23 9c 20 72    	m.nop
   268fa: 00 00        	s.nop16
   268fc: 00 00        	s.nop16
   268fe: 01 00        	c.nop
   26900: 00 00 00 00  	v.nop
   26904: 8c 1c 23 a0 20 72    	m.nop
   2690a: 00 00        	s.nop16
   2690c: 00 00        	s.nop16
   2690e: 3b 84        	<unknown>
   26910: 00 00 00 00  	v.nop
   26914: 0c 1d 23 a4 20 72    	m.nop
   2691a: 63 52 9f 03  	l.srli r9, r10, 0x1f
   2691e: 01 00        	c.nop
   26920: 00 00 00 00  	v.nop
   26924: 8c 1d 23 a8 20 72    	m.nop
   2692a: 00 00        	s.nop16
   2692c: 00 00        	s.nop16
   2692e: 01 00        	c.nop
   26930: 00 00 00 00  	v.nop
   26934: 0c 1e 23 ac 20 72    	m.nop
   2693a: 00 00        	s.nop16
   2693c: 00 00        	s.nop16
   2693e: 01 00        	c.nop
   26940: 00 00 00 00  	v.nop
   26944: 8c 1e 23 b0 20 72    	m.nop
   2694a: 00 00        	s.nop16
   2694c: 00 00        	s.nop16
   2694e: 01 00        	c.nop
   26950: 00 00 00 00  	v.nop
   26954: 0c 1f 23 b4 20 72    	m.nop
   2695a: 00 00        	s.nop16
   2695c: 00 00        	s.nop16
   2695e: 01 00        	c.nop
   26960: 00 00 00 00  	v.nop
   26964: 8c 1f 23 b8 20 72    	m.nop
   2696a: 00 00        	s.nop16
   2696c: 00 00        	s.nop16
   2696e: c3 84        	<unknown>
   26970: 00 00 00 00  	v.nop
   26974: 0c 20 23 bc 20 72    	m.nop
   2697a: 00 00        	s.nop16
   2697c: 00 00        	s.nop16
   2697e: 01 00        	c.nop
   26980: 00 00 00 00  	v.nop
   26984: 8c 20 23 c0 20 72    	m.nop
   2698a: 7b 00 6b 7f  	<unknown>
   2698e: 01 00        	c.nop
   26990: 30 32 99 88  	<unknown>
   26994: 00 00 00 00 00 00    	m.nop
   2699a: 39 01 09 00  	<unknown>
   2699e: 01 00        	c.nop
   269a0: 30 33 99 88  	<unknown>
   269a4: 00 00 00 00 00 00    	m.nop
   269aa: 00 00        	s.nop16
   269ac: 00 00        	s.nop16
   269ae: 01 00        	c.nop
   269b0: 30 34 99 88  	<unknown>
   269b4: 00 00 00 00 00 00    	m.nop
   269ba: 00 00        	s.nop16
   269bc: 00 00        	s.nop16
   269be: 01 00        	c.nop
   269c0: 30 35 99 88  	<unknown>
   269c4: 00 00 00 00 00 00    	m.nop
   269ca: 00 00        	s.nop16
   269cc: 00 00        	s.nop16
   269ce: 01 00        	c.nop
   269d0: 30 36 99 88  	<unknown>
   269d4: 00 00 00 00 00 00    	m.nop
   269da: 00 00        	s.nop16
   269dc: 00 00        	s.nop16
   269de: 01 00        	c.nop
   269e0: 30 37 99 88  	<unknown>
   269e4: 00 00 00 00 00 00    	m.nop
   269ea: 00 00        	s.nop16
   269ec: 00 00        	s.nop16
   269ee: 01 00        	c.nop
   269f0: 30 38 99 88  	<unknown>
   269f4: 00 00 00 00 00 00    	m.nop
   269fa: 00 00        	s.nop16
   269fc: 00 00        	s.nop16
   269fe: 01 00        	c.nop
   26a00: 30 39 99 88  	<unknown>
   26a04: 00 00 00 00 00 00    	m.nop
   26a0a: 1e 98        	s.tar.add ta19, ta0
   26a0c: 00 00        	s.nop16
   26a0e: 01 00        	c.nop
   26a10: 30 3a 99 88  	<unknown>
   26a14: 00 00 00 00 00 00    	m.nop
   26a1a: 00 00        	s.nop16
   26a1c: 00 00        	s.nop16
   26a1e: 01 00        	c.nop
   26a20: 30 3b 99 88  	<unknown>
   26a24: 00 00 00 00 00 00    	m.nop
   26a2a: 00 00        	s.nop16
   26a2c: 00 00        	s.nop16
   26a2e: 01 00        	c.nop
   26a30: 30 3c 99 88  	<unknown>
   26a34: 00 00 00 00 00 00    	m.nop
   26a3a: 00 00        	s.nop16
   26a3c: 00 00        	s.nop16
   26a3e: 01 00        	c.nop
   26a40: 30 3d 99 88  	<unknown>
   26a44: 00 00 00 00 00 00    	m.nop
   26a4a: 00 00        	s.nop16
   26a4c: 00 00        	s.nop16
   26a4e: 01 00        	c.nop
   26a50: 30 3e 99 88  	<unknown>
   26a54: 00 00 00 00 00 00    	m.nop
   26a5a: 00 00        	s.nop16
   26a5c: 00 00        	s.nop16
   26a5e: 01 00        	c.nop
   26a60: 30 3f 99 88  	<unknown>
   26a64: 00 00 00 00 00 00    	m.nop
   26a6a: 00 00        	s.nop16
   26a6c: 00 00        	s.nop16
   26a6e: 01 00        	c.nop
   26a70: 30 40 99 88  	<unknown>
   26a74: 00 00 00 00 00 00    	m.nop
   26a7a: 00 00        	s.nop16
   26a7c: 00 00        	s.nop16
   26a7e: 01 00        	c.nop
   26a80: 30 41 9b 88  	<unknown>
   26a84: 00 00 00 00 00 00    	m.nop
   26a8a: 00 00        	s.nop16
   26a8c: 00 00        	s.nop16
   26a8e: 01 00        	c.nop
   26a90: 00 00 00 00  	v.nop
   26a94: 00 00 00 00 00 00    	m.nop
   26a9a: 00 00        	s.nop16
   26a9c: 00 00        	s.nop16
   26a9e: fd 12        	<unknown>
   26aa0: 00 00 00 00  	v.nop
   26aa4: 00 00 00 00 00 00    	m.nop
   26aaa: fb 2f 20 7f  	<unknown>
   26aae: 01 00        	c.nop
   26ab0: 00 00 00 00  	v.nop
   26ab4: 00 00 00 00 00 00    	m.nop
   26aba: 31 12 50 06  	l.ld.w r8, [sp, 0x70]
   26abe: 01 00        	c.nop
   26ac0: 00 00 00 00  	v.nop
   26ac4: 00 00 00 00 00 00    	m.nop
   26aca: b1 12 58 1e  	l.ld.w r10, [sp, 0x1f8]
   26ace: 01 00        	c.nop
   26ad0: 00 00 00 00  	v.nop
   26ad4: 00 00 00 00 00 00    	m.nop
   26ada: 61 b9 c1 2f  	l.addi.s r5, r23, -0x3f
   26ade: 01 00        	c.nop
   26ae0: 00 00 00 00  	v.nop
   26ae4: 00 00 00 00 00 00    	m.nop
   26aea: 71 12 5c 1c  	l.ld.w r9, [sp, 0x1dc]
   26aee: 01 00        	c.nop
   26af0: 00 00 00 00  	v.nop
   26af4: 00 00 00 00 00 00    	m.nop
   26afa: 65 41 85 01  	l.andix r5, r8, 0x5
   26afe: 01 00        	c.nop
   26b00: 00 00 00 00  	v.nop
   26b04: 00 00 00 00 00 00    	m.nop
   26b0a: a3 51 08 18  	<unknown>
   26b0e: 01 00        	c.nop
   26b10: 00 00 00 00  	v.nop
   26b14: 00 00 00 00 00 00    	m.nop
   26b1a: 31 12 5c 08  	l.ld.w r8, [sp, 0x9c]
   26b1e: 01 00        	c.nop
   26b20: 00 00 00 00  	v.nop
   26b24: 00 00 00 00 00 00    	m.nop
   26b2a: 65 32 09 02  	l.orix r9, r6, 0x9
   26b2e: 01 00        	c.nop
   26b30: 00 00 00 00  	v.nop
   26b34: 00 00 00 00 00 00    	m.nop
   26b3a: a5 51 97 01  	l.andix r6, r10, 0x17
   26b3e: 71 45        	<unknown>
   26b40: 00 00 00 00  	v.nop
   26b44: 00 00 00 00 00 00    	m.nop
   26b4a: 00 00        	s.nop16
   26b4c: 00 00        	s.nop16
   26b4e: 2a 94        	<unknown>
   26b50: 00 00 00 00  	v.nop
   26b54: 00 00 00 00 00 00    	m.nop
   26b5a: 31 40 c7 00  	l.st.w r7, [r8]
   26b5e: 01 00        	c.nop
   26b60: 00 00 00 00  	v.nop
   26b64: 00 00 00 00 00 00    	m.nop
   26b6a: 31 47 cf fe  	l.st.w r15, [r8, -0x4]
   26b6e: 01 00        	c.nop
   26b70: 00 00 00 00  	v.nop
   26b74: 00 00 00 00 00 00    	m.nop
   26b7a: f1 11 4c 1e  	l.ld.w r7, [sp, 0x1ec]
   26b7e: 01 00        	c.nop
   26b80: 00 00 00 00  	v.nop
   26b84: 00 00 00 00 00 00    	m.nop
   26b8a: 31 42 c7 00  	l.st.w r7, [r8, 0x8]
   26b8e: 01 00        	c.nop
   26b90: 00 00 00 00  	v.nop
   26b94: 00 00 00 00 00 00    	m.nop
   26b9a: f1 11 40 1e  	l.ld.w r7, [sp, 0x1e0]
   26b9e: 01 00        	c.nop
   26ba0: 00 00 00 00  	v.nop
   26ba4: 00 00 00 00 00 00    	m.nop
   26baa: 31 41 c7 00  	l.st.w r7, [r8, 0x4]
   26bae: 01 00        	c.nop
   26bb0: 00 00 00 00  	v.nop
   26bb4: 00 00 00 00 00 00    	m.nop
   26bba: 2f 38 10 21  	l.ldi16.s r7, 0x200
   26bbe: 01 00        	c.nop
   26bc0: 00 00 00 00  	v.nop
   26bc4: 00 00 00 00 00 00    	m.nop
   26bca: 31 42 c7 0a  	l.st.w r7, [r8, 0xa8]
   26bce: 01 00        	c.nop
   26bd0: 00 00 00 00  	v.nop
   26bd4: 00 00 00 00 00 00    	m.nop
   26bda: 31 43 da 0a  	l.st.w r26, [r8, 0xac]
   26bde: 01 00        	c.nop
   26be0: 00 00 00 00  	v.nop
   26be4: 00 00 00 00 00 00    	m.nop
   26bea: 31 44 c6 0a  	l.st.w r6, [r8, 0xb0]
   26bee: 01 00        	c.nop
   26bf0: 00 00 00 00  	v.nop
   26bf4: 00 00 00 00 00 00    	m.nop
   26bfa: 31 45 da 0a  	l.st.w r26, [r8, 0xb4]
   26bfe: 01 00        	c.nop
   26c00: 00 00 00 00  	v.nop
   26c04: 00 00 00 00 00 00    	m.nop
   26c0a: 31 44 da 00  	l.st.w r26, [r8, 0x10]
   26c0e: 01 00        	c.nop
   26c10: 00 00 00 00  	v.nop
   26c14: 00 00 00 00 00 00    	m.nop
   26c1a: 31 40 cf 0e  	l.st.w r15, [r8, 0xe0]
   26c1e: 01 00        	c.nop
   26c20: 00 00 00 00  	v.nop
   26c24: 00 00 00 00 00 00    	m.nop
   26c2a: f1 11 4c 10  	l.ld.w r7, [sp, 0x10c]
   26c2e: 01 00        	c.nop
   26c30: 00 00 00 00  	v.nop
   26c34: 00 00 00 00 00 00    	m.nop
   26c3a: 31 41 c7 0e  	l.st.w r7, [r8, 0xe4]
   26c3e: 01 00        	c.nop
   26c40: 00 00 00 00  	v.nop
   26c44: 00 00 00 00 00 00    	m.nop
   26c4a: 31 42 c5 0e  	l.st.w r5, [r8, 0xe8]
   26c4e: 01 00        	c.nop
   26c50: 00 00 00 00  	v.nop
   26c54: 00 00 00 00 00 00    	m.nop
   26c5a: 31 42 cf 06  	l.st.w r15, [r8, 0x68]
   26c5e: 01 00        	c.nop
   26c60: 00 00 00 00  	v.nop
   26c64: 00 00 00 00 00 00    	m.nop
   26c6a: 31 44 de 0c  	l.st.w r30, [r8, 0xd0]
   26c6e: 01 00        	c.nop
   26c70: 00 00 00 00  	v.nop
   26c74: 00 00 00 00 00 00    	m.nop
   26c7a: f1 11 50 0a  	l.ld.w r7, [sp, 0xb0]
   26c7e: 01 00        	c.nop
   26c80: 00 00 00 00  	v.nop
   26c84: 00 00 00 00 00 00    	m.nop
   26c8a: 31 45 c7 0c  	l.st.w r7, [r8, 0xd4]
   26c8e: 01 00        	c.nop
   26c90: 00 00 00 00  	v.nop
   26c94: 00 00 00 00 00 00    	m.nop
   26c9a: f1 11 44 10  	l.ld.w r7, [sp, 0x104]
   26c9e: 01 00        	c.nop
   26ca0: 00 00 00 00  	v.nop
   26ca4: 00 00 00 00 00 00    	m.nop
   26caa: 31 46 c7 0c  	l.st.w r7, [r8, 0xd8]
   26cae: 01 00        	c.nop
   26cb0: 00 00 00 00  	v.nop
   26cb4: 00 00 00 00 00 00    	m.nop
   26cba: 2f 38 88 20  	l.ldi_hi.m1 r7, 0x100
   26cbe: 01 00        	c.nop
   26cc0: 00 00 00 00  	v.nop
   26cc4: 00 00 00 00 00 00    	m.nop
   26cca: 31 47 da 0c  	l.st.w r26, [r8, 0xdc]
   26cce: 01 00        	c.nop
   26cd0: 00 00 00 00  	v.nop
   26cd4: 00 00 00 00 00 00    	m.nop
   26cda: 7b 49 07 c0  	l.blt.u r9, r7, 0x5 <0x26d20>
   26cde: 01 00        	c.nop
   26ce0: 00 00 00 00  	v.nop
   26ce4: 00 00 00 00 00 00    	m.nop
   26cea: 2f 3a 10 21  	l.ldi16.s r7, 0x208
   26cee: 01 00        	c.nop
   26cf0: 00 00 00 00  	v.nop
   26cf4: 00 00 00 00 00 00    	m.nop
   26cfa: 2f 38 81 00  	l.ldi_hi r7, 0x20
   26cfe: 01 00        	c.nop
   26d00: 00 00 00 00  	v.nop
   26d04: 00 00 00 00 00 00    	m.nop
   26d0a: e1 39 0d 2f  	l.addi.s r7, r7, -0xf3
   26d0e: 01 00        	c.nop
   26d10: 00 00 00 00  	v.nop
   26d14: 00 00 00 00 00 00    	m.nop
   26d1a: 2b 10 80 01  	l.jal r0, 0x2 <0x26d30>
   26d1e: 01 00        	c.nop
   26d20: 00 00 00 00  	v.nop
   26d24: 00 00 00 00 00 00    	m.nop
   26d2a: e1 c9 01 00  	l.addi.u r7, r25, 0x1
   26d2e: 01 00        	c.nop
   26d30: 00 00 00 00  	v.nop
   26d34: 00 00 00 00 00 00    	m.nop
   26d3a: 71 13 5c 08  	l.ld.w r13, [sp, 0x9c]
   26d3e: 51 44        	<unknown>
   26d40: 00 00 00 00  	v.nop
   26d44: 00 00 00 00 00 00    	m.nop
   26d4a: 00 00        	s.nop16
   26d4c: 00 00        	s.nop16
   26d4e: 11 45        	<unknown>
   26d50: 00 00 00 00  	v.nop
   26d54: 00 00 00 00 00 00    	m.nop
   26d5a: 23 6a 08 10  	<unknown>
   26d5e: 01 00        	c.nop
   26d60: 00 00 00 00  	v.nop
   26d64: 00 00 00 00 00 00    	m.nop
   26d6a: 31 40 c7 00  	l.st.w r7, [r8]
   26d6e: 01 00        	c.nop
   26d70: 00 00 00 00  	v.nop
   26d74: 00 00 00 00 00 00    	m.nop
   26d7a: f3 b9 83 00  	l.movs r7, md23_wrk
   26d7e: 01 00        	c.nop
   26d80: 00 00 00 00  	v.nop
   26d84: 00 00 00 00 00 00    	m.nop
   26d8a: e5 39 8a 01  	l.andix r7, r7, 0xa
   26d8e: 01 00        	c.nop
   26d90: 00 00 00 00  	v.nop
   26d94: 00 00 00 00 00 00    	m.nop
   26d9a: fb 3b 00 00  	<unknown>
   26d9e: 01 00        	c.nop
   26da0: 00 00 00 00  	v.nop
   26da4: 00 00 00 00 00 00    	m.nop
   26daa: 2f 3a 04 21  	l.ldi16.s r7, 0x88
   26dae: 01 00        	c.nop
   26db0: 00 00 00 00  	v.nop
   26db4: 00 00 00 00 00 00    	m.nop
   26dba: e3 69 07 10  	<unknown>
   26dbe: 01 00        	c.nop
   26dc0: 00 00 00 00  	v.nop
   26dc4: 00 00 00 00 00 00    	m.nop
   26dca: b1 3a 40 00  	l.ld.w r10, [r7]
   26dce: 01 00        	c.nop
   26dd0: 00 00 00 00  	v.nop
   26dd4: 00 00 00 00 00 00    	m.nop
   26dda: f3 a2 83 00  	l.movs r11, md20_wrk
   26dde: 01 00        	c.nop
   26de0: 00 00 00 00  	v.nop
   26de4: 00 00 00 00 00 00    	m.nop
   26dea: 2f 70 f0 3e  	l.ldi_hi.m1 r14, 0xfe00
   26dee: 86 05        	<unknown>
   26df0: 00 00 00 00  	v.nop
   26df4: 00 00 00 00 00 00    	m.nop
   26dfa: ef 77 7f 1f  	l.ldi16.u r14, 0xffff
   26dfe: f9 8d        	<unknown>
   26e00: 00 00 00 00  	v.nop
   26e04: 00 00 00 00 00 00    	m.nop
   26e0a: 33 03 80 00  	l.movs r12, sip_id
   26e0e: 01 00        	c.nop
   26e10: 00 00 00 00  	v.nop
   26e14: 00 00 00 00 00 00    	m.nop
   26e1a: 6f 70 80 00  	l.ldi_hi r14, 0x1
   26e1e: 3e 06        	<unknown>
   26e20: 00 00 00 00  	v.nop
   26e24: 00 00 00 00 00 00    	m.nop
   26e2a: b1 13 4c 02  	l.ld.w r14, [sp, 0x2c]
   26e2e: 79 8d        	<unknown>
   26e30: 00 00 00 00  	v.nop
   26e34: 00 00 00 00 00 00    	m.nop
   26e3a: 00 00        	s.nop16
   26e3c: 00 00        	s.nop16
   26e3e: 4d 8d        	<unknown>
   26e40: 00 00 00 00  	v.nop
   26e44: 00 00 00 00 00 00    	m.nop
   26e4a: 00 00        	s.nop16
   26e4c: 00 00        	s.nop16
   26e4e: 79 8e        	<unknown>
   26e50: 00 00 00 00  	v.nop
   26e54: 00 00 00 00 00 00    	m.nop
   26e5a: 00 00        	s.nop16
   26e5c: 00 00        	s.nop16
   26e5e: 51 8d        	<unknown>
   26e60: 00 00 00 00  	v.nop
   26e64: 00 00 00 00 00 00    	m.nop
   26e6a: a5 52 1d 02  	l.orix r10, r10, 0x1d
   26e6e: 01 00        	c.nop
   26e70: 00 00 00 00  	v.nop
   26e74: 00 00 00 00 00 00    	m.nop
   26e7a: 31 38 ca 00  	l.st.w r10, [r7]
   26e7e: 01 00        	c.nop
   26e80: 00 00 00 00  	v.nop
   26e84: 00 00 00 00 00 00    	m.nop
   26e8a: 2f 3d 01 21  	l.ldi16.s r7, 0x34
   26e8e: 01 00        	c.nop
   26e90: 00 00 00 00  	v.nop
   26e94: 00 00 00 00 00 00    	m.nop
   26e9a: e3 69 07 10  	<unknown>
   26e9e: 01 00        	c.nop
   26ea0: 00 00 00 00  	v.nop
   26ea4: 00 00 00 00 00 00    	m.nop
   26eaa: b3 02 80 00  	l.movs r10, sip_id
   26eae: 01 00        	c.nop
   26eb0: 00 00 00 00  	v.nop
   26eb4: 00 00 00 00 00 00    	m.nop
   26eba: 31 38 da 00  	l.st.w r26, [r7]
   26ebe: 01 00        	c.nop
   26ec0: 00 00 00 00  	v.nop
   26ec4: 00 00 00 00 00 00    	m.nop
   26eca: b1 12 4c 06  	l.ld.w r10, [sp, 0x6c]
   26ece: 01 00        	c.nop
   26ed0: 00 00 00 00  	v.nop
   26ed4: 00 00 00 00 00 00    	m.nop
   26eda: 39 03 c0 01  	l.vclr.oa oacc12
   26ede: 01 00        	c.nop
   26ee0: 00 00 00 00  	v.nop
   26ee4: 00 00 00 00 00 00    	m.nop
   26eea: d3 00 fe 42  	l.vldi.64p0.oa  oacc3, 0x42fe
   26eee: 01 00        	c.nop
   26ef0: 00 00 00 00  	v.nop
   26ef4: 00 00 00 00 00 00    	m.nop
   26efa: 93 01 31 3f  	l.vldi.64p0.oa  oacc6, 0x3f31
   26efe: 01 00        	c.nop
   26f00: 00 00 00 00  	v.nop
   26f04: 00 00 00 00 00 00    	m.nop
   26f0a: 2d 50 20 00  	l.wait.dec r10
   26f0e: 01 00        	c.nop
   26f10: 00 00 00 00  	v.nop
   26f14: 00 00 00 00 00 00    	m.nop
   26f1a: 39 02 04 54  	<unknown>
   26f1e: 01 00        	c.nop
   26f20: 00 00 00 00  	v.nop
   26f24: 00 00 00 00 00 00    	m.nop
   26f2a: b1 12 54 14  	l.ld.w r10, [sp, 0x154]
   26f2e: 01 00        	c.nop
   26f30: 00 00 00 00  	v.nop
   26f34: 00 00 00 00 00 00    	m.nop
   26f3a: 91 01 18 72  	<unknown>
   26f3e: 01 00        	c.nop
   26f40: 00 00 00 00  	v.nop
   26f44: 00 00 00 00 00 00    	m.nop
   26f4a: 2f 58 08 11  	l.ldi16.u r11, 0x8100
   26f4e: 01 00        	c.nop
   26f50: 00 00 00 00  	v.nop
   26f54: 00 00 00 00 00 00    	m.nop
   26f5a: 2f 5d 80 00  	l.ldi_hi r11, 0x14
   26f5e: 01 00        	c.nop
   26f60: 00 00 00 00  	v.nop
   26f64: 00 00 00 00 00 00    	m.nop
   26f6a: 00 00        	s.nop16
   26f6c: 00 00        	s.nop16
   26f6e: 2e 95        	<unknown>
   26f70: 00 00 00 00  	v.nop
   26f74: 00 00 00 00 00 00    	m.nop
   26f7a: 39 05 0a 54  	<unknown>
   26f7e: 01 00        	c.nop
   26f80: 00 00 00 00  	v.nop
   26f84: 00 00 00 00 00 00    	m.nop
   26f8a: b1 12 54 1c  	l.ld.w r10, [sp, 0x1d4]
   26f8e: 01 00        	c.nop
   26f90: 00 00 00 00  	v.nop
   26f94: 00 00 00 00 00 00    	m.nop
   26f9a: 00 00        	s.nop16
   26f9c: 00 00        	s.nop16
   26f9e: c9 8c        	<unknown>
   26fa0: 00 00 00 00  	v.nop
   26fa4: 00 00 00 00 00 00    	m.nop
   26faa: b9 02 28 02  	l.vmov.va vacc10, vacc8
   26fae: 01 00        	c.nop
   26fb0: 00 00 00 00  	v.nop
   26fb4: 0c 01 00 08 a0 4a    	m.nop
   26fba: 00 00        	s.nop16
   26fbc: 00 00        	s.nop16
   26fbe: 01 00        	c.nop
   26fc0: 00 00 00 00  	v.nop
   26fc4: 10 02 05 20 80 6a    	m.msf.ln oacc8, oacc8
   26fca: 00 00        	s.nop16
   26fcc: 00 00        	s.nop16
   26fce: 01 00        	c.nop
   26fd0: 00 00 00 00  	v.nop
   26fd4: 50 02 05 24 80 6a    	m.msf.ln oacc9, oacc9
   26fda: 00 00        	s.nop16
   26fdc: 00 00        	s.nop16
   26fde: 01 00        	c.nop
   26fe0: 00 00 00 00  	v.nop
   26fe4: 90 02 05 28 80 6a    	m.msf.ln oacc10, oacc10
   26fea: 00 00        	s.nop16
   26fec: 00 00        	s.nop16
   26fee: 01 00        	c.nop
   26ff0: 00 00 00 00  	v.nop
   26ff4: d0 02 05 2c 80 6a    	m.msf.ln oacc11, oacc11
   26ffa: 00 00        	s.nop16
   26ffc: 00 00        	s.nop16
   26ffe: 01 00        	c.nop
   27000: 00 00 00 00  	v.nop
   27004: 0c 01 16 10 20 76    	m.nop
   2700a: 00 00        	s.nop16
   2700c: 00 00        	s.nop16
   2700e: 01 00        	c.nop
   27010: 00 00 00 00  	v.nop
   27014: c8 00 03 10 a0 1a    	m.nop
   2701a: 00 00        	s.nop16
   2701c: 00 00        	s.nop16
   2701e: 01 00        	c.nop
   27020: 00 00 00 00  	v.nop
   27024: 98 00 03 18 a0 6a    	m.nop
   2702a: 00 00        	s.nop16
   2702c: 00 00        	s.nop16
   2702e: 01 00        	c.nop
   27030: 00 00 00 00  	v.nop
   27034: 08 01 04 28 90 0a    	m.nop
   2703a: 00 00        	s.nop16
   2703c: 00 00        	s.nop16
   2703e: 01 00        	c.nop
   27040: 34 02 84 92  	<unknown>
   27044: 00 00 00 00 00 00    	m.nop
   2704a: 00 00        	s.nop16
   2704c: 00 00        	s.nop16
   2704e: 01 00        	c.nop
   27050: 00 00 00 00  	v.nop
   27054: 00 00 00 00 00 00    	m.nop
   2705a: 33 f8 98 00  	<unknown>
   2705e: 01 00        	c.nop
   27060: 00 00 00 00  	v.nop
   27064: 00 00 00 00 00 00    	m.nop
   2706a: 31 3a c4 fe  	l.st.w r4, [r7, -0x18]
   2706e: 01 00        	c.nop
   27070: 00 00 00 00  	v.nop
   27074: 00 00 00 00 00 00    	m.nop
   2707a: 31 39 cf fe  	l.st.w r15, [r7, -0x1c]
   2707e: 01 00        	c.nop
   27080: 00 00 00 00  	v.nop
   27084: 00 00 00 00 00 00    	m.nop
   2708a: 31 11 50 1c  	l.ld.w r4, [sp, 0x1d0]
   2708e: 01 00        	c.nop
   27090: 00 00 00 00  	v.nop
   27094: 00 00 00 00 00 00    	m.nop
   2709a: 31 3c c4 fe  	l.st.w r4, [r7, -0x10]
   2709e: 01 00        	c.nop
   270a0: 00 00 00 00  	v.nop
   270a4: 00 00 00 00 00 00    	m.nop
   270aa: 31 11 40 1c  	l.ld.w r4, [sp, 0x1c0]
   270ae: 01 00        	c.nop
   270b0: 00 00 00 00  	v.nop
   270b4: 00 00 00 00 00 00    	m.nop
   270ba: 31 3b c4 fe  	l.st.w r4, [r7, -0x14]
   270be: 01 00        	c.nop
   270c0: 00 00 00 00  	v.nop
   270c4: 00 00 00 00 00 00    	m.nop
   270ca: 31 3c c6 08  	l.st.w r6, [r7, 0x90]
   270ce: 01 00        	c.nop
   270d0: 00 00 00 00  	v.nop
   270d4: 00 00 00 00 00 00    	m.nop
   270da: 31 3d da 08  	l.st.w r26, [r7, 0x94]
   270de: 01 00        	c.nop
   270e0: 00 00 00 00  	v.nop
   270e4: 00 00 00 00 00 00    	m.nop
   270ea: 31 3e da 08  	l.st.w r26, [r7, 0x98]
   270ee: 01 00        	c.nop
   270f0: 00 00 00 00  	v.nop
   270f4: 00 00 00 00 00 00    	m.nop
   270fa: 31 3f da 08  	l.st.w r26, [r7, 0x9c]
   270fe: 01 00        	c.nop
   27100: 00 00 00 00  	v.nop
   27104: 00 00 00 00 00 00    	m.nop
   2710a: 31 3e da fe  	l.st.w r26, [r7, -0x8]
   2710e: 01 00        	c.nop
   27110: 00 00 00 00  	v.nop
   27114: 00 00 00 00 00 00    	m.nop
   2711a: 31 3a c5 0c  	l.st.w r5, [r7, 0xc8]
   2711e: 01 00        	c.nop
   27120: 00 00 00 00  	v.nop
   27124: 00 00 00 00 00 00    	m.nop
   2712a: 31 11 4c 10  	l.ld.w r4, [sp, 0x10c]
   2712e: 01 00        	c.nop
   27130: 00 00 00 00  	v.nop
   27134: 00 00 00 00 00 00    	m.nop
   2713a: 31 3b c4 0c  	l.st.w r4, [r7, 0xcc]
   2713e: 01 00        	c.nop
   27140: 00 00 00 00  	v.nop
   27144: 00 00 00 00 00 00    	m.nop
   2714a: 31 11 58 1c  	l.ld.w r4, [sp, 0x1d8]
   2714e: 01 00        	c.nop
   27150: 00 00 00 00  	v.nop
   27154: 00 00 00 00 00 00    	m.nop
   2715a: 31 3c c4 0c  	l.st.w r4, [r7, 0xd0]
   2715e: 01 00        	c.nop
   27160: 00 00 00 00  	v.nop
   27164: 00 00 00 00 00 00    	m.nop
   2716a: 31 3c cf 04  	l.st.w r15, [r7, 0x50]
   2716e: 01 00        	c.nop
   27170: 00 00 00 00  	v.nop
   27174: 00 00 00 00 00 00    	m.nop
   2717a: 31 11 44 10  	l.ld.w r4, [sp, 0x104]
   2717e: 01 00        	c.nop
   27180: 00 00 00 00  	v.nop
   27184: 00 00 00 00 00 00    	m.nop
   2718a: 31 3e c4 0a  	l.st.w r4, [r7, 0xb8]
   2718e: 01 00        	c.nop
   27190: 00 00 00 00  	v.nop
   27194: 00 00 00 00 00 00    	m.nop
   2719a: 31 11 50 0a  	l.ld.w r4, [sp, 0xb0]
   2719e: 01 00        	c.nop
   271a0: 00 00 00 00  	v.nop
   271a4: 00 00 00 00 00 00    	m.nop
   271aa: 31 3f c4 0a  	l.st.w r4, [r7, 0xbc]
   271ae: 01 00        	c.nop
   271b0: 00 00 00 00  	v.nop
   271b4: 00 00 00 00 00 00    	m.nop
   271ba: 31 11 5c 1a  	l.ld.w r4, [sp, 0x1bc]
   271be: 01 00        	c.nop
   271c0: 00 00 00 00  	v.nop
   271c4: 00 00 00 00 00 00    	m.nop
   271ca: 31 38 c4 0c  	l.st.w r4, [r7, 0xc0]
   271ce: 01 00        	c.nop
   271d0: 00 00 00 00  	v.nop
   271d4: 00 00 00 00 00 00    	m.nop
   271da: 2f 20 88 20  	l.ldi_hi.m1 r4, 0x100
   271de: 01 00        	c.nop
   271e0: 00 00 00 00  	v.nop
   271e4: 00 00 00 00 00 00    	m.nop
   271ea: 31 39 da 0c  	l.st.w r26, [r7, 0xc4]
   271ee: 01 00        	c.nop
   271f0: 00 00 00 00  	v.nop
   271f4: 00 00 00 00 00 00    	m.nop
   271fa: 7b 49 04 c0  	l.blt.u r9, r4, 0x5 <0x27240>
   271fe: 01 00        	c.nop
   27200: 00 00 00 00  	v.nop
   27204: 00 00 00 00 00 00    	m.nop
   2720a: 2f 22 10 21  	l.ldi16.s r4, 0x208
   2720e: 01 00        	c.nop
   27210: 00 00 00 00  	v.nop
   27214: 00 00 00 00 00 00    	m.nop
   2721a: 2f 20 81 00  	l.ldi_hi r4, 0x20
   2721e: 01 00        	c.nop
   27220: 00 00 00 00  	v.nop
   27224: 00 00 00 00 00 00    	m.nop
   2722a: 00 00        	s.nop16
   2722c: 00 00        	s.nop16
   2722e: 35 02        	<unknown>
   27230: 00 00 00 00  	v.nop
   27234: 00 00 00 00 00 00    	m.nop
   2723a: 2b 10 80 01  	l.jal r0, 0x2 <0x27250>
   2723e: 01 00        	c.nop
   27240: 00 00 00 00  	v.nop
   27244: 00 00 00 00 00 00    	m.nop
   2724a: 21 c9 01 01  	l.addi.u r4, r25, 0x101
   2724e: 01 00        	c.nop
   27250: 00 00 00 00  	v.nop
   27254: 00 00 00 00 00 00    	m.nop
   2725a: 31 40 c4 00  	l.st.w r4, [r8]
   2725e: 91 42        	<unknown>
   27260: 00 00 00 00  	v.nop
   27264: 00 00 00 00 00 00    	m.nop
   2726a: 33 b9 83 00  	l.movs r4, md23_wrk
   2726e: 01 00        	c.nop
   27270: 00 00 00 00  	v.nop
   27274: 00 00 00 00 00 00    	m.nop
   2727a: 25 21 85 01  	l.andix r4, r4, 0x5
   2727e: 01 00        	c.nop
   27280: 00 00 00 00  	v.nop
   27284: 00 00 00 00 00 00    	m.nop
   2728a: bb 20 00 40  	<unknown>
   2728e: 01 00        	c.nop
   27290: 00 00 00 00  	v.nop
   27294: 00 00 00 00 00 00    	m.nop
   2729a: 2b 28 d3 3d  	l.jal r0, -0xacb <0x1c5e0>
   2729e: 01 00        	c.nop
   272a0: 00 00 00 00  	v.nop
   272a4: 00 00 00 00 00 00    	m.nop
   272aa: 31 11 5c 08  	l.ld.w r4, [sp, 0x9c]
   272ae: 01 00        	c.nop
   272b0: 00 00 00 00  	v.nop
   272b4: 00 00 00 00 00 00    	m.nop
   272ba: 2f 2a 04 21  	l.ldi16.s r5, 0x88
   272be: 01 00        	c.nop
   272c0: 00 00 00 00  	v.nop
   272c4: 00 00 00 00 00 00    	m.nop
   272ca: 2f 48 f0 3e  	l.ldi_hi.m1 r9, 0xfe00
   272ce: 01 00        	c.nop
   272d0: 00 00 00 00  	v.nop
   272d4: 00 00 00 00 00 00    	m.nop
   272da: 00 00        	s.nop16
   272dc: 00 00        	s.nop16
   272de: 16 92        	<unknown>
   272e0: 00 00 00 00  	v.nop
   272e4: 00 00 00 00 00 00    	m.nop
   272ea: 71 21 40 00  	l.ld.w r5, [r4]
   272ee: 01 00        	c.nop
   272f0: 00 00 00 00  	v.nop
   272f4: 00 00 00 00 00 00    	m.nop
   272fa: b3 a1 83 00  	l.movs r6, md20_wrk
   272fe: 01 00        	c.nop
   27300: 00 00 00 00  	v.nop
   27304: 00 00 00 00 00 00    	m.nop
   2730a: 33 02 80 00  	l.movs r8, sip_id
   2730e: 06 03        	<unknown>
   27310: 00 00 00 00  	v.nop
   27314: 00 00 00 00 00 00    	m.nop
   2731a: a5 31 89 01  	l.andix r6, r6, 0x9
   2731e: 3e 04        	<unknown>
   27320: 00 00 00 00  	v.nop
   27324: 00 00 00 00 00 00    	m.nop
   2732a: ef 4f 7f 1f  	l.ldi16.u r9, 0xffff
   2732e: 01 00        	c.nop
   27330: 00 00 00 00  	v.nop
   27334: 00 00 00 00 00 00    	m.nop
   2733a: 6f 48 80 00  	l.ldi_hi r9, 0x1
   2733e: 01 00        	c.nop
   27340: 00 00 00 00  	v.nop
   27344: 00 00 00 00 00 00    	m.nop
   2734a: 65 29 89 01  	l.andix r5, r5, 0x9
   2734e: 01 00        	c.nop
   27350: 00 00 00 00  	v.nop
   27354: 00 00 00 00 00 00    	m.nop
   2735a: 71 12 4c 02  	l.ld.w r9, [sp, 0x2c]
   2735e: 01 00        	c.nop
   27360: 00 00 00 00  	v.nop
   27364: 00 00 00 00 00 00    	m.nop
   2736a: 65 29 06 02  	l.orix r5, r5, 0x6
   2736e: 01 00        	c.nop
   27370: 00 00 00 00  	v.nop
   27374: 00 00 00 00 00 00    	m.nop
   2737a: 00 00        	s.nop16
   2737c: 00 00        	s.nop16
   2737e: 65 8c        	<unknown>
   27380: 00 00 00 00  	v.nop
   27384: 00 00 00 00 00 00    	m.nop
   2738a: 65 29 08 02  	l.orix r5, r5, 0x8
   2738e: 01 00        	c.nop
   27390: 00 00 00 00  	v.nop
   27394: 00 00 00 00 00 00    	m.nop
   2739a: 65 29 1d 02  	l.orix r5, r5, 0x1d
   2739e: 01 00        	c.nop
   273a0: 00 00 00 00  	v.nop
   273a4: 00 00 00 00 00 00    	m.nop
   273aa: 31 20 c5 00  	l.st.w r5, [r4]
   273ae: 01 00        	c.nop
   273b0: 00 00 00 00  	v.nop
   273b4: 00 00 00 00 00 00    	m.nop
   273ba: 2b 18 d2 3d  	l.jal r0, -0xadd <0x1c5e0>
   273be: 01 00        	c.nop
   273c0: 00 00 00 00  	v.nop
   273c4: 00 00 00 00 00 00    	m.nop
   273ca: 00 00        	s.nop16
   273cc: 00 00        	s.nop16
   273ce: 05 43        	<unknown>
   273d0: 00 00 00 00  	v.nop
   273d4: 00 00 00 00 00 00    	m.nop
   273da: f1 11 5c 38  	l.ld.w r7, [sp, 0x39c]
   273de: 01 00        	c.nop
   273e0: 00 00 00 00  	v.nop
   273e4: 00 00 00 00 00 00    	m.nop
   273ea: 31 12 50 38  	l.ld.w r8, [sp, 0x390]
   273ee: 01 00        	c.nop
   273f0: 00 00 00 00  	v.nop
   273f4: 00 00 00 00 00 00    	m.nop
   273fa: 71 12 54 38  	l.ld.w r9, [sp, 0x394]
   273fe: 01 00        	c.nop
   27400: 00 00 00 00  	v.nop
   27404: 00 00 00 00 00 00    	m.nop
   2740a: 2f 60 00 25  	l.ldi16.s r12, 0x2000
   2740e: 01 00        	c.nop
   27410: 00 00 00 00  	v.nop
   27414: 00 00 00 00 00 00    	m.nop
   2741a: b1 12 5c 36  	l.ld.w r10, [sp, 0x37c]
   2741e: 01 00        	c.nop
   27420: 00 00 00 00  	v.nop
   27424: 00 00 00 00 00 00    	m.nop
   2742a: a3 39 06 18  	<unknown>
   2742e: 01 00        	c.nop
   27430: 00 00 00 00  	v.nop
   27434: 00 00 00 00 00 00    	m.nop
   2743a: e3 41 19 03  	l.slli r7, r8, 0x19
   2743e: 01 00        	c.nop
   27440: 00 00 00 00  	v.nop
   27444: 00 00 00 00 00 00    	m.nop
   2744a: 23 4a 04 03  	l.slli r8, r9, 0x4
   2744e: 01 00        	c.nop
   27450: 00 00 00 00  	v.nop
   27454: 00 00 00 00 00 00    	m.nop
   2745a: ef 4f 7f 1f  	l.ldi16.u r9, 0xffff
   2745e: 01 00        	c.nop
   27460: 00 00 00 00  	v.nop
   27464: 00 00 00 00 00 00    	m.nop
   2746a: ef 4f 83 00  	l.ldi_hi r9, 0x7f
   2746e: 01 00        	c.nop
   27470: 00 00 00 00  	v.nop
   27474: 00 00 00 00 00 00    	m.nop
   2747a: 6f 67 91 08  	l.ldi_hi r12, 0x423d
   2747e: a6 85        	<unknown>
   27480: 00 00 00 00  	v.nop
   27484: 00 00 00 00 00 00    	m.nop
   2748a: 65 32 8b 01  	l.andix r9, r6, 0xb
   2748e: 32 94        	<unknown>
   27490: 00 00 00 00  	v.nop
   27494: 00 00 00 00 00 00    	m.nop
   2749a: 31 17 c6 38  	l.st.w r6, [sp, 0x39c]
   2749e: 01 00        	c.nop
   274a0: 00 00 00 00  	v.nop
   274a4: 00 00 00 00 00 00    	m.nop
   274aa: 2f 36 33 2b  	l.ldi16.s r6, 0x5678
   274ae: 01 00        	c.nop
   274b0: 00 00 00 00  	v.nop
   274b4: 00 00 00 00 00 00    	m.nop
   274ba: 2f 35 91 02  	l.ldi_hi r6, 0x1234
   274be: 01 00        	c.nop
   274c0: 00 00 00 00  	v.nop
   274c4: 00 00 00 00 00 00    	m.nop
   274ca: 31 40 c6 00  	l.st.w r6, [r8]
   274ce: 01 00        	c.nop
   274d0: 00 00 00 00  	v.nop
   274d4: 00 00 00 00 00 00    	m.nop
   274da: a5 39 0b 02  	l.orix r6, r7, 0xb
   274de: 01 00        	c.nop
   274e0: 00 00 00 00  	v.nop
   274e4: 00 00 00 00 00 00    	m.nop
   274ea: ef 5f 7f 1f  	l.ldi16.u r11, 0xffff
   274ee: 01 00        	c.nop
   274f0: 00 00 00 00  	v.nop
   274f4: 00 00 00 00 00 00    	m.nop
   274fa: ef 5f ff 00  	l.ldi_hi r11, 0xfff
   274fe: 01 00        	c.nop
   27500: 00 00 00 00  	v.nop
   27504: 00 00 00 00 00 00    	m.nop
   2750a: 31 41 cb 00  	l.st.w r11, [r8, 0x4]
   2750e: 01 00        	c.nop
   27510: 00 00 00 00  	v.nop
   27514: 00 00 00 00 00 00    	m.nop
   2751a: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   2751e: 01 00        	c.nop
   27520: 00 00 00 00  	v.nop
   27524: 00 00 00 00 00 00    	m.nop
   2752a: a5 39 09 02  	l.orix r6, r7, 0x9
   2752e: 01 00        	c.nop
   27530: 00 00 00 00  	v.nop
   27534: 00 00 00 00 00 00    	m.nop
   2753a: 31 40 ca 00  	l.st.w r10, [r8]
   2753e: 01 00        	c.nop
   27540: 00 00 00 00  	v.nop
   27544: 00 00 00 00 00 00    	m.nop
   2754a: 31 41 c5 00  	l.st.w r5, [r8, 0x4]
   2754e: 01 00        	c.nop
   27550: 00 00 00 00  	v.nop
   27554: 00 00 00 00 00 00    	m.nop
   2755a: 31 42 c6 00  	l.st.w r6, [r8, 0x8]
   2755e: 01 00        	c.nop
   27560: 00 00 00 00  	v.nop
   27564: 00 00 00 00 00 00    	m.nop
   2756a: f1 11 0c 3c  	l.ld.b r7, [sp, 0x3cc]
   2756e: 01 00        	c.nop
   27570: 00 00 00 00  	v.nop
   27574: 00 00 00 00 00 00    	m.nop
   2757a: a3 a1 99 18  	<unknown>
   2757e: 01 00        	c.nop
   27580: 00 00 00 00  	v.nop
   27584: 00 00 00 00 00 00    	m.nop
   2758a: e5 39 9c 01  	l.andix r7, r7, 0x1c
   2758e: 01 00        	c.nop
   27590: 00 00 00 00  	v.nop
   27594: 00 00 00 00 00 00    	m.nop
   2759a: bb 38 00 00  	<unknown>
   2759e: 01 00        	c.nop
   275a0: 00 00 00 00  	v.nop
   275a4: 00 00 00 00 00 00    	m.nop
   275aa: 2b 48 fc 3d  	l.jal r0, -0x837 <0x1f230>
   275ae: 01 00        	c.nop
   275b0: 00 00 00 00  	v.nop
   275b4: 00 00 00 00 00 00    	m.nop
   275ba: 2b 18 ff 3d  	l.jal r0, -0x80d <0x1f4e0>
   275be: 01 00        	c.nop
   275c0: 00 00 00 00  	v.nop
   275c4: 00 00 00 00 00 00    	m.nop
   275ca: 31 11 44 0a  	l.ld.w r4, [sp, 0xa4]
   275ce: 01 00        	c.nop
   275d0: 00 00 00 00  	v.nop
   275d4: 00 00 00 00 00 00    	m.nop
   275da: 7b 22 80 c0  	l.blt.s r4, r0, 0x9 <0x27660>
   275de: 01 00        	c.nop
   275e0: 00 00 00 00  	v.nop
   275e4: 00 00 00 00 00 00    	m.nop
   275ea: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   275ee: 01 00        	c.nop
   275f0: 00 00 00 00  	v.nop
   275f4: 00 00 00 00 00 00    	m.nop
   275fa: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   275fe: 01 00        	c.nop
   27600: 00 00 00 00  	v.nop
   27604: 00 00 00 00 00 00    	m.nop
   2760a: 31 21 40 00  	l.ld.w r4, [r4]
   2760e: 01 00        	c.nop
   27610: 00 00 00 00  	v.nop
   27614: 00 00 00 00 00 00    	m.nop
   2761a: 71 11 48 16  	l.ld.w r5, [sp, 0x168]
   2761e: 01 00        	c.nop
   27620: 00 00 00 00  	v.nop
   27624: 00 00 00 00 00 00    	m.nop
   2762a: 25 21 85 02  	l.xorix r4, r4, 0x5
   2762e: 01 00        	c.nop
   27630: 00 00 00 00  	v.nop
   27634: 00 00 00 00 00 00    	m.nop
   2763a: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   2763e: 01 00        	c.nop
   27640: 00 00 00 00  	v.nop
   27644: 00 00 00 00 00 00    	m.nop
   2764a: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   2764e: 01 00        	c.nop
   27650: 00 00 00 00  	v.nop
   27654: 00 00 00 00 00 00    	m.nop
   2765a: 31 28 c4 00  	l.st.w r4, [r5]
   2765e: 01 00        	c.nop
   27660: 00 00 00 00  	v.nop
   27664: 00 00 00 00 00 00    	m.nop
   2766a: 31 12 4c 04  	l.ld.w r8, [sp, 0x4c]
   2766e: 01 00        	c.nop
   27670: 00 00 00 00  	v.nop
   27674: 00 00 00 00 00 00    	m.nop
   2767a: 71 12 5c 16  	l.ld.w r9, [sp, 0x17c]
   2767e: 01 00        	c.nop
   27680: 00 00 00 00  	v.nop
   27684: 00 00 00 00 00 00    	m.nop
   2768a: f1 12 5c 0c  	l.ld.w r11, [sp, 0xdc]
   2768e: 01 00        	c.nop
   27690: 00 00 00 00  	v.nop
   27694: 00 00 00 00 00 00    	m.nop
   2769a: b1 12 58 16  	l.ld.w r10, [sp, 0x178]
   2769e: 01 00        	c.nop
   276a0: 00 00 00 00  	v.nop
   276a4: 00 00 00 00 00 00    	m.nop
   276aa: 71 11 40 0a  	l.ld.w r5, [sp, 0xa0]
   276ae: 01 00        	c.nop
   276b0: 00 00 00 00  	v.nop
   276b4: 00 00 00 00 00 00    	m.nop
   276ba: 2f 38 80 30  	l.ldi_hi.m1 r7, 0x8000
   276be: 01 00        	c.nop
   276c0: 00 00 00 00  	v.nop
   276c4: 00 00 00 00 00 00    	m.nop
   276ca: fb 29 80 c0  	l.blt.s r5, r0, 0x7 <0x27730>
   276ce: 01 00        	c.nop
   276d0: 00 00 00 00  	v.nop
   276d4: 00 00 00 00 00 00    	m.nop
   276da: 00 00        	s.nop16
   276dc: 00 00        	s.nop16
   276de: 7d 42        	<unknown>
   276e0: 00 00 00 00  	v.nop
   276e4: 00 00 00 00 00 00    	m.nop
   276ea: 3b 25 a5 c0  	l.blt.s r4, r5, 0x34 <0x27a20>
   276ee: 01 00        	c.nop
   276f0: 00 00 00 00  	v.nop
   276f4: 00 00 00 00 00 00    	m.nop
   276fa: 31 41 40 00  	l.ld.w r4, [r8]
   276fe: 01 00        	c.nop
   27700: 00 00 00 00  	v.nop
   27704: 00 00 00 00 00 00    	m.nop
   2770a: 63 39 85 02  	l.srl     r5, r7, r5
   2770e: 01 00        	c.nop
   27710: 00 00 00 00  	v.nop
   27714: 00 00 00 00 00 00    	m.nop
   2771a: 25 21 85 02  	l.xorix r4, r4, 0x5
   2771e: 01 00        	c.nop
   27720: 00 00 00 00  	v.nop
   27724: 00 00 00 00 00 00    	m.nop
   2772a: 31 40 c4 00  	l.st.w r4, [r8]
   2772e: 01 00        	c.nop
   27730: 00 00 00 00  	v.nop
   27734: 00 00 00 00 00 00    	m.nop
   2773a: 7b 52 80 c0  	l.blt.s r10, r0, 0x9 <0x277c0>
   2773e: 01 00        	c.nop
   27740: 00 00 00 00  	v.nop
   27744: 00 00 00 00 00 00    	m.nop
   2774a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   2774e: 01 00        	c.nop
   27750: 00 00 00 00  	v.nop
   27754: 00 00 00 00 00 00    	m.nop
   2775a: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   2775e: 01 00        	c.nop
   27760: 00 00 00 00  	v.nop
   27764: 00 00 00 00 00 00    	m.nop
   2776a: 31 21 40 00  	l.ld.w r4, [r4]
   2776e: 01 00        	c.nop
   27770: 00 00 00 00  	v.nop
   27774: 00 00 00 00 00 00    	m.nop
   2777a: 71 11 4c 16  	l.ld.w r5, [sp, 0x16c]
   2777e: 01 00        	c.nop
   27780: 00 00 00 00  	v.nop
   27784: 00 00 00 00 00 00    	m.nop
   2778a: 25 21 85 02  	l.xorix r4, r4, 0x5
   2778e: 01 00        	c.nop
   27790: 00 00 00 00  	v.nop
   27794: 00 00 00 00 00 00    	m.nop
   2779a: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   2779e: 01 00        	c.nop
   277a0: 00 00 00 00  	v.nop
   277a4: 00 00 00 00 00 00    	m.nop
   277aa: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   277ae: 01 00        	c.nop
   277b0: 00 00 00 00  	v.nop
   277b4: 00 00 00 00 00 00    	m.nop
   277ba: 31 28 c4 00  	l.st.w r4, [r5]
   277be: 01 00        	c.nop
   277c0: 00 00 00 00  	v.nop
   277c4: 00 00 00 00 00 00    	m.nop
   277ca: 71 11 40 0e  	l.ld.w r5, [sp, 0xe0]
   277ce: 7d 55        	<unknown>
   277d0: 00 00 00 00  	v.nop
   277d4: 00 00 00 00 00 00    	m.nop
   277da: 31 13 4c 06  	l.ld.w r12, [sp, 0x6c]
   277de: 01 00        	c.nop
   277e0: 00 00 00 00  	v.nop
   277e4: 00 00 00 00 00 00    	m.nop
   277ea: fb 29 80 c0  	l.blt.s r5, r0, 0x7 <0x27850>
   277ee: 01 00        	c.nop
   277f0: 00 00 00 00  	v.nop
   277f4: 00 00 00 00 00 00    	m.nop
   277fa: 00 00        	s.nop16
   277fc: 00 00        	s.nop16
   277fe: 7d 42        	<unknown>
   27800: 00 00 00 00  	v.nop
   27804: 00 00 00 00 00 00    	m.nop
   2780a: 7b 22 a5 c0  	l.blt.s r4, r5, 0x29 <0x27a90>
   2780e: 01 00        	c.nop
   27810: 00 00 00 00  	v.nop
   27814: 00 00 00 00 00 00    	m.nop
   2781a: 31 41 40 00  	l.ld.w r4, [r8]
   2781e: 01 00        	c.nop
   27820: 00 00 00 00  	v.nop
   27824: 00 00 00 00 00 00    	m.nop
   2782a: 63 39 85 02  	l.srl     r5, r7, r5
   2782e: 01 00        	c.nop
   27830: 00 00 00 00  	v.nop
   27834: 00 00 00 00 00 00    	m.nop
   2783a: 25 21 85 02  	l.xorix r4, r4, 0x5
   2783e: 01 00        	c.nop
   27840: 00 00 00 00  	v.nop
   27844: 00 00 00 00 00 00    	m.nop
   2784a: 31 40 c4 00  	l.st.w r4, [r8]
   2784e: 01 00        	c.nop
   27850: 00 00 00 00  	v.nop
   27854: 00 00 00 00 00 00    	m.nop
   2785a: 31 11 50 5e  	l.ld.w r4, [sp, 0x5f0]
   2785e: 01 00        	c.nop
   27860: 00 00 00 00  	v.nop
   27864: 00 00 00 00 00 00    	m.nop
   2786a: bb 22 a0 c0  	l.blt.s r4, r0, 0x2a <0x27b00>
   2786e: 01 00        	c.nop
   27870: 00 00 00 00  	v.nop
   27874: 00 00 00 00 00 00    	m.nop
   2787a: 31 11 40 60  	l.ld.w r4, [sp, 0x600]
   2787e: 01 00        	c.nop
   27880: 00 00 00 00  	v.nop
   27884: 00 00 00 00 00 00    	m.nop
   2788a: 3b 22 1a 40  	<unknown>
   2788e: 01 00        	c.nop
   27890: 00 00 00 00  	v.nop
   27894: 00 00 00 00 00 00    	m.nop
   2789a: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   2789e: 01 00        	c.nop
   278a0: 00 00 00 00  	v.nop
   278a4: 00 00 00 00 00 00    	m.nop
   278aa: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   278ae: 01 00        	c.nop
   278b0: 00 00 00 00  	v.nop
   278b4: 00 00 00 00 00 00    	m.nop
   278ba: 31 11 48 60  	l.ld.w r4, [sp, 0x608]
   278be: 16 83        	<unknown>
   278c0: 00 00 00 00  	v.nop
   278c4: 00 00 00 00 00 00    	m.nop
   278ca: 71 31 40 00  	l.ld.w r5, [r6]
   278ce: 01 00        	c.nop
   278d0: 00 00 00 00  	v.nop
   278d4: 00 00 00 00 00 00    	m.nop
   278da: 23 39 84 02  	l.srl     r4, r7, r4
   278de: 01 00        	c.nop
   278e0: 00 00 00 00  	v.nop
   278e4: 00 00 00 00 00 00    	m.nop
   278ea: 25 29 84 02  	l.xorix r4, r5, 0x4
   278ee: 01 00        	c.nop
   278f0: 00 00 00 00  	v.nop
   278f4: 00 00 00 00 00 00    	m.nop
   278fa: 31 30 c4 00  	l.st.w r4, [r6]
   278fe: 01 00        	c.nop
   27900: 00 00 00 00  	v.nop
   27904: 00 00 00 00 00 00    	m.nop
   2790a: 31 14 ca 5e  	l.st.w r10, [sp, 0x5f0]
   2790e: 01 00        	c.nop
   27910: 00 00 00 00  	v.nop
   27914: 00 00 00 00 00 00    	m.nop
   2791a: 31 11 50 7e  	l.ld.w r4, [sp, 0x7f0]
   2791e: 01 00        	c.nop
   27920: 00 00 00 00  	v.nop
   27924: 00 00 00 00 00 00    	m.nop
   2792a: 3b 20 a0 80  	l.bge.s r4, r0, 0x20 <0x27b20>
   2792e: 01 00        	c.nop
   27930: 00 00 00 00  	v.nop
   27934: 00 00 00 00 00 00    	m.nop
   2793a: 31 11 40 36  	l.ld.w r4, [sp, 0x360]
   2793e: 01 00        	c.nop
   27940: 00 00 00 00  	v.nop
   27944: 00 00 00 00 00 00    	m.nop
   2794a: bb 54 a4 80  	l.bge.s r10, r4, 0x32 <0x27c60>
   2794e: 01 00        	c.nop
   27950: 00 00 00 00  	v.nop
   27954: 00 00 00 00 00 00    	m.nop
   2795a: 31 11 50 36  	l.ld.w r4, [sp, 0x370]
   2795e: 01 00        	c.nop
   27960: 00 00 00 00  	v.nop
   27964: 00 00 00 00 00 00    	m.nop
   2796a: 3b 22 1a 40  	<unknown>
   2796e: 01 00        	c.nop
   27970: 00 00 00 00  	v.nop
   27974: 00 00 00 00 00 00    	m.nop
   2797a: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   2797e: 01 00        	c.nop
   27980: 00 00 00 00  	v.nop
   27984: 00 00 00 00 00 00    	m.nop
   2798a: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   2798e: 01 00        	c.nop
   27990: 00 00 00 00  	v.nop
   27994: 00 00 00 00 00 00    	m.nop
   2799a: 31 11 58 36  	l.ld.w r4, [sp, 0x378]
   2799e: 16 83        	<unknown>
   279a0: 00 00 00 00  	v.nop
   279a4: 00 00 00 00 00 00    	m.nop
   279aa: 71 31 40 00  	l.ld.w r5, [r6]
   279ae: 01 00        	c.nop
   279b0: 00 00 00 00  	v.nop
   279b4: 00 00 00 00 00 00    	m.nop
   279ba: 23 39 84 02  	l.srl     r4, r7, r4
   279be: 01 00        	c.nop
   279c0: 00 00 00 00  	v.nop
   279c4: 00 00 00 00 00 00    	m.nop
   279ca: 25 29 84 02  	l.xorix r4, r5, 0x4
   279ce: 01 00        	c.nop
   279d0: 00 00 00 00  	v.nop
   279d4: 00 00 00 00 00 00    	m.nop
   279da: 31 30 c4 00  	l.st.w r4, [r6]
   279de: 01 00        	c.nop
   279e0: 00 00 00 00  	v.nop
   279e4: 00 00 00 00 00 00    	m.nop
   279ea: 31 10 ca 36  	l.st.w r10, [sp, 0x360]
   279ee: 01 00        	c.nop
   279f0: 00 00 00 00  	v.nop
   279f4: 00 00 00 00 00 00    	m.nop
   279fa: 31 11 40 56  	l.ld.w r4, [sp, 0x560]
   279fe: 01 00        	c.nop
   27a00: 00 00 00 00  	v.nop
   27a04: 00 00 00 00 00 00    	m.nop
   27a0a: 3b 22 a0 80  	l.bge.s r4, r0, 0x28 <0x27c80>
   27a0e: 01 00        	c.nop
   27a10: 00 00 00 00  	v.nop
   27a14: 00 00 00 00 00 00    	m.nop
   27a1a: 2b 28 83 01  	l.jal r0, 0x35 <0x27d60>
   27a1e: 01 00        	c.nop
   27a20: 00 00 00 00  	v.nop
   27a24: 00 00 00 00 00 00    	m.nop
   27a2a: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   27a2e: 81 12        	<unknown>
   27a30: 00 00 00 00  	v.nop
   27a34: 00 00 00 00 00 00    	m.nop
   27a3a: 63 39 85 02  	l.srl     r5, r7, r5
   27a3e: 01 00        	c.nop
   27a40: 00 00 00 00  	v.nop
   27a44: 00 00 00 00 00 00    	m.nop
   27a4a: 31 31 40 00  	l.ld.w r4, [r6]
   27a4e: 01 00        	c.nop
   27a50: 00 00 00 00  	v.nop
   27a54: 00 00 00 00 00 00    	m.nop
   27a5a: 25 21 85 02  	l.xorix r4, r4, 0x5
   27a5e: 01 00        	c.nop
   27a60: 00 00 00 00  	v.nop
   27a64: 00 00 00 00 00 00    	m.nop
   27a6a: 31 30 c4 00  	l.st.w r4, [r6]
   27a6e: 01 00        	c.nop
   27a70: 00 00 00 00  	v.nop
   27a74: 00 00 00 00 00 00    	m.nop
   27a7a: 7b 53 c0 bf  	l.bge.s r10, r0, -0x33 <0x27740>
   27a7e: 01 00        	c.nop
   27a80: 00 00 00 00  	v.nop
   27a84: 00 00 00 00 00 00    	m.nop
   27a8a: 2b 20 fd 3f  	l.jal r0, -0x2c <0x277c0>
   27a8e: 01 00        	c.nop
   27a90: 00 00 00 00  	v.nop
   27a94: 00 00 00 00 00 00    	m.nop
   27a9a: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   27a9e: 81 12        	<unknown>
   27aa0: 00 00 00 00  	v.nop
   27aa4: 00 00 00 00 00 00    	m.nop
   27aaa: 63 39 85 02  	l.srl     r5, r7, r5
   27aae: 01 00        	c.nop
   27ab0: 00 00 00 00  	v.nop
   27ab4: 00 00 00 00 00 00    	m.nop
   27aba: 31 31 40 00  	l.ld.w r4, [r6]
   27abe: 01 00        	c.nop
   27ac0: 00 00 00 00  	v.nop
   27ac4: 00 00 00 00 00 00    	m.nop
   27aca: 25 21 85 02  	l.xorix r4, r4, 0x5
   27ace: 01 00        	c.nop
   27ad0: 00 00 00 00  	v.nop
   27ad4: 00 00 00 00 00 00    	m.nop
   27ada: 31 30 c4 00  	l.st.w r4, [r6]
   27ade: 01 00        	c.nop
   27ae0: 00 00 00 00  	v.nop
   27ae4: 00 00 00 00 00 00    	m.nop
   27aea: 31 11 50 5e  	l.ld.w r4, [sp, 0x5f0]
   27aee: 01 00        	c.nop
   27af0: 00 00 00 00  	v.nop
   27af4: 00 00 00 00 00 00    	m.nop
   27afa: 3b 26 c0 bf  	l.bge.s r4, r0, -0x28 <0x27870>
   27afe: 01 00        	c.nop
   27b00: 00 00 00 00  	v.nop
   27b04: 00 00 00 00 00 00    	m.nop
   27b0a: 31 11 50 7e  	l.ld.w r4, [sp, 0x7f0]
   27b0e: 01 00        	c.nop
   27b10: 00 00 00 00  	v.nop
   27b14: 00 00 00 00 00 00    	m.nop
   27b1a: bb 20 e0 ff  	l.blt.s r4, r0, -0x1e <0x27930>
   27b1e: 01 00        	c.nop
   27b20: 00 00 00 00  	v.nop
   27b24: 00 00 00 00 00 00    	m.nop
   27b2a: 2f 21 40 21  	l.ldi16.s r4, 0x804
   27b2e: fd 42        	<unknown>
   27b30: 00 00 00 00  	v.nop
   27b34: 00 00 00 00 00 00    	m.nop
   27b3a: 23 11 04 18  	<unknown>
   27b3e: 01 00        	c.nop
   27b40: 00 00 00 00  	v.nop
   27b44: 00 00 00 00 00 00    	m.nop
   27b4a: 31 21 40 00  	l.ld.w r4, [r4]
   27b4e: 01 00        	c.nop
   27b50: 00 00 00 00  	v.nop
   27b54: 00 00 00 00 00 00    	m.nop
   27b5a: 7b 2a 84 c0  	l.blt.s r5, r4, 0x9 <0x27be0>
   27b5e: 01 00        	c.nop
   27b60: 00 00 00 00  	v.nop
   27b64: 00 00 00 00 00 00    	m.nop
   27b6a: 71 41 40 00  	l.ld.w r5, [r8]
   27b6e: 01 00        	c.nop
   27b70: 00 00 00 00  	v.nop
   27b74: 00 00 00 00 00 00    	m.nop
   27b7a: 23 39 84 02  	l.srl     r4, r7, r4
   27b7e: 01 00        	c.nop
   27b80: 00 00 00 00  	v.nop
   27b84: 00 00 00 00 00 00    	m.nop
   27b8a: 25 29 84 02  	l.xorix r4, r5, 0x4
   27b8e: 01 00        	c.nop
   27b90: 00 00 00 00  	v.nop
   27b94: 00 00 00 00 00 00    	m.nop
   27b9a: 31 40 c4 00  	l.st.w r4, [r8]
   27b9e: 01 00        	c.nop
   27ba0: 00 00 00 00  	v.nop
   27ba4: 00 00 00 00 00 00    	m.nop
   27baa: 31 14 ca 7e  	l.st.w r10, [sp, 0x7f0]
   27bae: 01 00        	c.nop
   27bb0: 00 00 00 00  	v.nop
   27bb4: 00 00 00 00 00 00    	m.nop
   27bba: 31 11 40 36  	l.ld.w r4, [sp, 0x360]
   27bbe: 01 00        	c.nop
   27bc0: 00 00 00 00  	v.nop
   27bc4: 00 00 00 00 00 00    	m.nop
   27bca: 7b 56 c4 ff  	l.blt.s r10, r4, -0x27 <0x27950>
   27bce: 01 00        	c.nop
   27bd0: 00 00 00 00  	v.nop
   27bd4: 00 00 00 00 00 00    	m.nop
   27bda: 2b 48 80 01  	l.jal r0, 0x9 <0x27c60>
   27bde: 01 00        	c.nop
   27be0: 00 00 00 00  	v.nop
   27be4: 00 00 00 00 00 00    	m.nop
   27bea: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   27bee: 01 12        	<unknown>
   27bf0: 00 00 00 00  	v.nop
   27bf4: 00 00 00 00 00 00    	m.nop
   27bfa: 23 39 84 02  	l.srl     r4, r7, r4
   27bfe: 01 00        	c.nop
   27c00: 00 00 00 00  	v.nop
   27c04: 00 00 00 00 00 00    	m.nop
   27c0a: 71 31 40 00  	l.ld.w r5, [r6]
   27c0e: 01 00        	c.nop
   27c10: 00 00 00 00  	v.nop
   27c14: 00 00 00 00 00 00    	m.nop
   27c1a: 25 29 84 02  	l.xorix r4, r5, 0x4
   27c1e: 01 00        	c.nop
   27c20: 00 00 00 00  	v.nop
   27c24: 00 00 00 00 00 00    	m.nop
   27c2a: 31 30 c4 00  	l.st.w r4, [r6]
   27c2e: 01 00        	c.nop
   27c30: 00 00 00 00  	v.nop
   27c34: 00 00 00 00 00 00    	m.nop
   27c3a: 31 14 ca 7e  	l.st.w r10, [sp, 0x7f0]
   27c3e: 01 00        	c.nop
   27c40: 00 00 00 00  	v.nop
   27c44: 00 00 00 00 00 00    	m.nop
   27c4a: 31 11 40 36  	l.ld.w r4, [sp, 0x360]
   27c4e: 01 00        	c.nop
   27c50: 00 00 00 00  	v.nop
   27c54: 00 00 00 00 00 00    	m.nop
   27c5a: 3b 54 c4 ff  	l.blt.s r10, r4, -0x30 <0x27950>
   27c5e: 01 00        	c.nop
   27c60: 00 00 00 00  	v.nop
   27c64: 00 00 00 00 00 00    	m.nop
   27c6a: 31 11 40 56  	l.ld.w r4, [sp, 0x560]
   27c6e: 01 00        	c.nop
   27c70: 00 00 00 00  	v.nop
   27c74: 00 00 00 00 00 00    	m.nop
   27c7a: fb 23 80 c0  	l.blt.s r4, r0, 0xf <0x27d60>
   27c7e: 01 00        	c.nop
   27c80: 00 00 00 00  	v.nop
   27c84: 00 00 00 00 00 00    	m.nop
   27c8a: 31 11 54 56  	l.ld.w r4, [sp, 0x574]
   27c8e: 01 00        	c.nop
   27c90: 00 00 00 00  	v.nop
   27c94: 00 00 00 00 00 00    	m.nop
   27c9a: 2f 28 01 21  	l.ldi16.s r5, 0x20
   27c9e: 01 00        	c.nop
   27ca0: 00 00 00 00  	v.nop
   27ca4: 00 00 00 00 00 00    	m.nop
   27caa: bb 21 85 80  	l.bge.s r4, r5, 0x6 <0x27d00>
   27cae: 01 00        	c.nop
   27cb0: 00 00 00 00  	v.nop
   27cb4: 00 00 00 00 00 00    	m.nop
   27cba: 71 41 40 00  	l.ld.w r5, [r8]
   27cbe: 01 00        	c.nop
   27cc0: 00 00 00 00  	v.nop
   27cc4: 00 00 00 00 00 00    	m.nop
   27cca: 23 39 84 02  	l.srl     r4, r7, r4
   27cce: 01 00        	c.nop
   27cd0: 00 00 00 00  	v.nop
   27cd4: 00 00 00 00 00 00    	m.nop
   27cda: 25 29 84 02  	l.xorix r4, r5, 0x4
   27cde: 01 00        	c.nop
   27ce0: 00 00 00 00  	v.nop
   27ce4: 00 00 00 00 00 00    	m.nop
   27cea: 31 40 c4 00  	l.st.w r4, [r8]
   27cee: 01 00        	c.nop
   27cf0: 00 00 00 00  	v.nop
   27cf4: 00 00 00 00 00 00    	m.nop
   27cfa: 2b 30 80 01  	l.jal r0, 0x6 <0x27d50>
   27cfe: 01 00        	c.nop
   27d00: 00 00 00 00  	v.nop
   27d04: 00 00 00 00 00 00    	m.nop
   27d0a: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   27d0e: 01 12        	<unknown>
   27d10: 00 00 00 00  	v.nop
   27d14: 00 00 00 00 00 00    	m.nop
   27d1a: 23 39 84 02  	l.srl     r4, r7, r4
   27d1e: 01 00        	c.nop
   27d20: 00 00 00 00  	v.nop
   27d24: 00 00 00 00 00 00    	m.nop
   27d2a: 71 31 40 00  	l.ld.w r5, [r6]
   27d2e: 01 00        	c.nop
   27d30: 00 00 00 00  	v.nop
   27d34: 00 00 00 00 00 00    	m.nop
   27d3a: 25 29 84 02  	l.xorix r4, r5, 0x4
   27d3e: 01 00        	c.nop
   27d40: 00 00 00 00  	v.nop
   27d44: 00 00 00 00 00 00    	m.nop
   27d4a: 31 30 c4 00  	l.st.w r4, [r6]
   27d4e: 01 00        	c.nop
   27d50: 00 00 00 00  	v.nop
   27d54: 00 00 00 00 00 00    	m.nop
   27d5a: 31 10 ca 56  	l.st.w r10, [sp, 0x560]
   27d5e: 01 00        	c.nop
   27d60: 00 00 00 00  	v.nop
   27d64: 00 00 00 00 00 00    	m.nop
   27d6a: 2f 24 58 21  	l.ldi16.s r4, 0xb10
   27d6e: 01 00        	c.nop
   27d70: 00 00 00 00  	v.nop
   27d74: 00 00 00 00 00 00    	m.nop
   27d7a: 23 11 04 18  	<unknown>
   27d7e: 01 00        	c.nop
   27d80: 00 00 00 00  	v.nop
   27d84: 00 00 00 00 00 00    	m.nop
   27d8a: 31 21 40 00  	l.ld.w r4, [r4]
   27d8e: 01 00        	c.nop
   27d90: 00 00 00 00  	v.nop
   27d94: 00 00 00 00 00 00    	m.nop
   27d9a: 7b 54 84 80  	l.bge.s r10, r4, 0x11 <0x27ea0>
   27d9e: 01 00        	c.nop
   27da0: 00 00 00 00  	v.nop
   27da4: 00 00 00 00 00 00    	m.nop
   27daa: 2f 20 59 21  	l.ldi16.s r4, 0xb20
   27dae: 01 00        	c.nop
   27db0: 00 00 00 00  	v.nop
   27db4: 00 00 00 00 00 00    	m.nop
   27dba: 23 11 04 18  	<unknown>
   27dbe: 01 00        	c.nop
   27dc0: 00 00 00 00  	v.nop
   27dc4: 00 00 00 00 00 00    	m.nop
   27dca: 31 21 40 00  	l.ld.w r4, [r4]
   27dce: 01 00        	c.nop
   27dd0: 00 00 00 00  	v.nop
   27dd4: 00 00 00 00 00 00    	m.nop
   27dda: bb 22 1a 40  	<unknown>
   27dde: 01 00        	c.nop
   27de0: 00 00 00 00  	v.nop
   27de4: 00 00 00 00 00 00    	m.nop
   27dea: 2f 22 59 21  	l.ldi16.s r4, 0xb28
   27dee: 01 00        	c.nop
   27df0: 00 00 00 00  	v.nop
   27df4: 00 00 00 00 00 00    	m.nop
   27dfa: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   27dfe: 01 00        	c.nop
   27e00: 00 00 00 00  	v.nop
   27e04: 00 00 00 00 00 00    	m.nop
   27e0a: 23 11 04 18  	<unknown>
   27e0e: 01 00        	c.nop
   27e10: 00 00 00 00  	v.nop
   27e14: 00 00 00 00 00 00    	m.nop
   27e1a: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   27e1e: 01 00        	c.nop
   27e20: 00 00 00 00  	v.nop
   27e24: 00 00 00 00 00 00    	m.nop
   27e2a: 31 21 40 00  	l.ld.w r4, [r4]
   27e2e: 16 83        	<unknown>
   27e30: 00 00 00 00  	v.nop
   27e34: 00 00 00 00 00 00    	m.nop
   27e3a: 71 31 40 00  	l.ld.w r5, [r6]
   27e3e: 01 00        	c.nop
   27e40: 00 00 00 00  	v.nop
   27e44: 00 00 00 00 00 00    	m.nop
   27e4a: 23 39 84 02  	l.srl     r4, r7, r4
   27e4e: 01 00        	c.nop
   27e50: 00 00 00 00  	v.nop
   27e54: 00 00 00 00 00 00    	m.nop
   27e5a: 25 29 84 02  	l.xorix r4, r5, 0x4
   27e5e: 01 00        	c.nop
   27e60: 00 00 00 00  	v.nop
   27e64: 00 00 00 00 00 00    	m.nop
   27e6a: 31 30 c4 00  	l.st.w r4, [r6]
   27e6e: 01 00        	c.nop
   27e70: 00 00 00 00  	v.nop
   27e74: 00 00 00 00 00 00    	m.nop
   27e7a: 2f 24 58 21  	l.ldi16.s r4, 0xb10
   27e7e: 01 00        	c.nop
   27e80: 00 00 00 00  	v.nop
   27e84: 00 00 00 00 00 00    	m.nop
   27e8a: 23 11 04 18  	<unknown>
   27e8e: 01 00        	c.nop
   27e90: 00 00 00 00  	v.nop
   27e94: 00 00 00 00 00 00    	m.nop
   27e9a: 31 20 ca 00  	l.st.w r10, [r4]
   27e9e: 01 00        	c.nop
   27ea0: 00 00 00 00  	v.nop
   27ea4: 00 00 00 00 00 00    	m.nop
   27eaa: 2f 24 68 21  	l.ldi16.s r4, 0xd10
   27eae: 01 00        	c.nop
   27eb0: 00 00 00 00  	v.nop
   27eb4: 00 00 00 00 00 00    	m.nop
   27eba: 23 11 04 18  	<unknown>
   27ebe: 01 00        	c.nop
   27ec0: 00 00 00 00  	v.nop
   27ec4: 00 00 00 00 00 00    	m.nop
   27eca: 31 21 40 00  	l.ld.w r4, [r4]
   27ece: 01 00        	c.nop
   27ed0: 00 00 00 00  	v.nop
   27ed4: 00 00 00 00 00 00    	m.nop
   27eda: bb 24 80 c0  	l.blt.s r4, r0, 0x12 <0x27ff0>
   27ede: 01 00        	c.nop
   27ee0: 00 00 00 00  	v.nop
   27ee4: 00 00 00 00 00 00    	m.nop
   27eea: 2f 21 69 21  	l.ldi16.s r4, 0xd24
   27eee: fd 42        	<unknown>
   27ef0: 00 00 00 00  	v.nop
   27ef4: 00 00 00 00 00 00    	m.nop
   27efa: 23 11 04 18  	<unknown>
   27efe: 01 00        	c.nop
   27f00: 00 00 00 00  	v.nop
   27f04: 00 00 00 00 00 00    	m.nop
   27f0a: 31 21 40 00  	l.ld.w r4, [r4]
   27f0e: 01 00        	c.nop
   27f10: 00 00 00 00  	v.nop
   27f14: 00 00 00 00 00 00    	m.nop
   27f1a: bb 29 84 c0  	l.blt.s r5, r4, 0x6 <0x27f70>
   27f1e: 01 00        	c.nop
   27f20: 00 00 00 00  	v.nop
   27f24: 00 00 00 00 00 00    	m.nop
   27f2a: 71 41 40 00  	l.ld.w r5, [r8]
   27f2e: 01 00        	c.nop
   27f30: 00 00 00 00  	v.nop
   27f34: 00 00 00 00 00 00    	m.nop
   27f3a: 23 39 84 02  	l.srl     r4, r7, r4
   27f3e: 01 00        	c.nop
   27f40: 00 00 00 00  	v.nop
   27f44: 00 00 00 00 00 00    	m.nop
   27f4a: 25 29 84 02  	l.xorix r4, r5, 0x4
   27f4e: 01 00        	c.nop
   27f50: 00 00 00 00  	v.nop
   27f54: 00 00 00 00 00 00    	m.nop
   27f5a: 31 40 c4 00  	l.st.w r4, [r8]
   27f5e: 01 00        	c.nop
   27f60: 00 00 00 00  	v.nop
   27f64: 00 00 00 00 00 00    	m.nop
   27f6a: 2b 30 80 01  	l.jal r0, 0x6 <0x27fc0>
   27f6e: 01 00        	c.nop
   27f70: 00 00 00 00  	v.nop
   27f74: 00 00 00 00 00 00    	m.nop
   27f7a: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   27f7e: 01 12        	<unknown>
   27f80: 00 00 00 00  	v.nop
   27f84: 00 00 00 00 00 00    	m.nop
   27f8a: 23 39 84 02  	l.srl     r4, r7, r4
   27f8e: 01 00        	c.nop
   27f90: 00 00 00 00  	v.nop
   27f94: 00 00 00 00 00 00    	m.nop
   27f9a: 71 31 40 00  	l.ld.w r5, [r6]
   27f9e: 01 00        	c.nop
   27fa0: 00 00 00 00  	v.nop
   27fa4: 00 00 00 00 00 00    	m.nop
   27faa: 25 29 84 02  	l.xorix r4, r5, 0x4
   27fae: 01 00        	c.nop
   27fb0: 00 00 00 00  	v.nop
   27fb4: 00 00 00 00 00 00    	m.nop
   27fba: 31 30 c4 00  	l.st.w r4, [r6]
   27fbe: 01 00        	c.nop
   27fc0: 00 00 00 00  	v.nop
   27fc4: 00 00 00 00 00 00    	m.nop
   27fca: 2f 24 68 21  	l.ldi16.s r4, 0xd10
   27fce: 01 00        	c.nop
   27fd0: 00 00 00 00  	v.nop
   27fd4: 00 00 00 00 00 00    	m.nop
   27fda: 23 11 04 18  	<unknown>
   27fde: 01 00        	c.nop
   27fe0: 00 00 00 00  	v.nop
   27fe4: 00 00 00 00 00 00    	m.nop
   27fea: 31 20 ca 00  	l.st.w r10, [r4]
   27fee: 01 00        	c.nop
   27ff0: 00 00 00 00  	v.nop
   27ff4: 00 00 00 00 00 00    	m.nop
   27ffa: 27 11 a2 09  	l.addi_sft.s.sft0 r4, sp, 0x22
   27ffe: 01 00        	c.nop
   28000: 00 00 00 00  	v.nop
   28004: 00 00 00 00 00 00    	m.nop
   2800a: 31 21 40 00  	l.ld.w r4, [r4]
   2800e: 01 00        	c.nop
   28010: 00 00 00 00  	v.nop
   28014: 00 00 00 00 00 00    	m.nop
   2801a: 3b 54 84 80  	l.bge.s r10, r4, 0x10 <0x28110>
   2801e: 01 00        	c.nop
   28020: 00 00 00 00  	v.nop
   28024: 00 00 00 00 00 00    	m.nop
   2802a: 2f 24 44 21  	l.ldi16.s r4, 0x890
   2802e: 01 00        	c.nop
   28030: 00 00 00 00  	v.nop
   28034: 00 00 00 00 00 00    	m.nop
   2803a: 23 11 04 18  	<unknown>
   2803e: 01 00        	c.nop
   28040: 00 00 00 00  	v.nop
   28044: 00 00 00 00 00 00    	m.nop
   2804a: 31 21 40 00  	l.ld.w r4, [r4]
   2804e: 01 00        	c.nop
   28050: 00 00 00 00  	v.nop
   28054: 00 00 00 00 00 00    	m.nop
   2805a: bb 22 1a 40  	<unknown>
   2805e: 01 00        	c.nop
   28060: 00 00 00 00  	v.nop
   28064: 00 00 00 00 00 00    	m.nop
   2806a: 2f 26 44 21  	l.ldi16.s r4, 0x898
   2806e: 01 00        	c.nop
   28070: 00 00 00 00  	v.nop
   28074: 00 00 00 00 00 00    	m.nop
   2807a: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   2807e: 01 00        	c.nop
   28080: 00 00 00 00  	v.nop
   28084: 00 00 00 00 00 00    	m.nop
   2808a: 23 11 04 18  	<unknown>
   2808e: 01 00        	c.nop
   28090: 00 00 00 00  	v.nop
   28094: 00 00 00 00 00 00    	m.nop
   2809a: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   2809e: 01 00        	c.nop
   280a0: 00 00 00 00  	v.nop
   280a4: 00 00 00 00 00 00    	m.nop
   280aa: 31 21 40 00  	l.ld.w r4, [r4]
   280ae: 16 83        	<unknown>
   280b0: 00 00 00 00  	v.nop
   280b4: 00 00 00 00 00 00    	m.nop
   280ba: 71 31 40 00  	l.ld.w r5, [r6]
   280be: 01 00        	c.nop
   280c0: 00 00 00 00  	v.nop
   280c4: 00 00 00 00 00 00    	m.nop
   280ca: 23 39 84 02  	l.srl     r4, r7, r4
   280ce: 01 00        	c.nop
   280d0: 00 00 00 00  	v.nop
   280d4: 00 00 00 00 00 00    	m.nop
   280da: 25 29 84 02  	l.xorix r4, r5, 0x4
   280de: 01 00        	c.nop
   280e0: 00 00 00 00  	v.nop
   280e4: 00 00 00 00 00 00    	m.nop
   280ea: 31 30 c4 00  	l.st.w r4, [r6]
   280ee: 01 00        	c.nop
   280f0: 00 00 00 00  	v.nop
   280f4: 00 00 00 00 00 00    	m.nop
   280fa: 27 11 a2 09  	l.addi_sft.s.sft0 r4, sp, 0x22
   280fe: 01 00        	c.nop
   28100: 00 00 00 00  	v.nop
   28104: 00 00 00 00 00 00    	m.nop
   2810a: 31 20 ca 00  	l.st.w r10, [r4]
   2810e: 01 00        	c.nop
   28110: 00 00 00 00  	v.nop
   28114: 00 00 00 00 00 00    	m.nop
   2811a: 27 11 aa 09  	l.addi_sft.s.sft0 r4, sp, 0x2a
   2811e: 01 00        	c.nop
   28120: 00 00 00 00  	v.nop
   28124: 00 00 00 00 00 00    	m.nop
   2812a: 31 21 40 00  	l.ld.w r4, [r4]
   2812e: 01 00        	c.nop
   28130: 00 00 00 00  	v.nop
   28134: 00 00 00 00 00 00    	m.nop
   2813a: bb 24 80 c0  	l.blt.s r4, r0, 0x12 <0x28250>
   2813e: 01 00        	c.nop
   28140: 00 00 00 00  	v.nop
   28144: 00 00 00 00 00 00    	m.nop
   2814a: 2f 25 54 21  	l.ldi16.s r4, 0xa94
   2814e: 01 00        	c.nop
   28150: 00 00 00 00  	v.nop
   28154: 00 00 00 00 00 00    	m.nop
   2815a: 23 11 04 18  	<unknown>
   2815e: 01 00        	c.nop
   28160: 00 00 00 00  	v.nop
   28164: 00 00 00 00 00 00    	m.nop
   2816a: 31 21 40 00  	l.ld.w r4, [r4]
   2816e: 01 00        	c.nop
   28170: 00 00 00 00  	v.nop
   28174: 00 00 00 00 00 00    	m.nop
   2817a: 2f 28 01 21  	l.ldi16.s r5, 0x20
   2817e: 01 00        	c.nop
   28180: 00 00 00 00  	v.nop
   28184: 00 00 00 00 00 00    	m.nop
   2818a: bb 21 85 80  	l.bge.s r4, r5, 0x6 <0x281e0>
   2818e: 01 00        	c.nop
   28190: 00 00 00 00  	v.nop
   28194: 00 00 00 00 00 00    	m.nop
   2819a: 71 41 40 00  	l.ld.w r5, [r8]
   2819e: 01 00        	c.nop
   281a0: 00 00 00 00  	v.nop
   281a4: 00 00 00 00 00 00    	m.nop
   281aa: 23 39 84 02  	l.srl     r4, r7, r4
   281ae: 01 00        	c.nop
   281b0: 00 00 00 00  	v.nop
   281b4: 00 00 00 00 00 00    	m.nop
   281ba: 25 29 84 02  	l.xorix r4, r5, 0x4
   281be: 01 00        	c.nop
   281c0: 00 00 00 00  	v.nop
   281c4: 00 00 00 00 00 00    	m.nop
   281ca: 31 40 c4 00  	l.st.w r4, [r8]
   281ce: 01 00        	c.nop
   281d0: 00 00 00 00  	v.nop
   281d4: 00 00 00 00 00 00    	m.nop
   281da: 2b 30 80 01  	l.jal r0, 0x6 <0x28230>
   281de: 01 00        	c.nop
   281e0: 00 00 00 00  	v.nop
   281e4: 00 00 00 00 00 00    	m.nop
   281ea: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   281ee: 01 12        	<unknown>
   281f0: 00 00 00 00  	v.nop
   281f4: 00 00 00 00 00 00    	m.nop
   281fa: 23 39 84 02  	l.srl     r4, r7, r4
   281fe: 01 00        	c.nop
   28200: 00 00 00 00  	v.nop
   28204: 00 00 00 00 00 00    	m.nop
   2820a: 71 31 40 00  	l.ld.w r5, [r6]
   2820e: 01 00        	c.nop
   28210: 00 00 00 00  	v.nop
   28214: 00 00 00 00 00 00    	m.nop
   2821a: 25 29 84 02  	l.xorix r4, r5, 0x4
   2821e: 01 00        	c.nop
   28220: 00 00 00 00  	v.nop
   28224: 00 00 00 00 00 00    	m.nop
   2822a: 31 30 c4 00  	l.st.w r4, [r6]
   2822e: 01 00        	c.nop
   28230: 00 00 00 00  	v.nop
   28234: 00 00 00 00 00 00    	m.nop
   2823a: 27 11 aa 09  	l.addi_sft.s.sft0 r4, sp, 0x2a
   2823e: 01 00        	c.nop
   28240: 00 00 00 00  	v.nop
   28244: 00 00 00 00 00 00    	m.nop
   2824a: 31 20 ca 00  	l.st.w r10, [r4]
   2824e: 01 00        	c.nop
   28250: 00 00 00 00  	v.nop
   28254: 00 00 00 00 00 00    	m.nop
   2825a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   2825e: 01 00        	c.nop
   28260: 00 00 00 00  	v.nop
   28264: 00 00 00 00 00 00    	m.nop
   2826a: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   2826e: 01 00        	c.nop
   28270: 00 00 00 00  	v.nop
   28274: 00 00 00 00 00 00    	m.nop
   2827a: fb 52 8b 80  	l.bge.s r10, r11, 0xb <0x28320>
   2827e: 01 00        	c.nop
   28280: 00 00 00 00  	v.nop
   28284: 00 00 00 00 00 00    	m.nop
   2828a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   2828e: 01 00        	c.nop
   28290: 00 00 00 00  	v.nop
   28294: 00 00 00 00 00 00    	m.nop
   2829a: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   2829e: 01 00        	c.nop
   282a0: 00 00 00 00  	v.nop
   282a4: 00 00 00 00 00 00    	m.nop
   282aa: 31 21 40 00  	l.ld.w r4, [r4]
   282ae: 01 00        	c.nop
   282b0: 00 00 00 00  	v.nop
   282b4: 00 00 00 00 00 00    	m.nop
   282ba: 71 11 50 16  	l.ld.w r5, [sp, 0x170]
   282be: 01 00        	c.nop
   282c0: 00 00 00 00  	v.nop
   282c4: 00 00 00 00 00 00    	m.nop
   282ca: 25 21 85 02  	l.xorix r4, r4, 0x5
   282ce: 01 00        	c.nop
   282d0: 00 00 00 00  	v.nop
   282d4: 00 00 00 00 00 00    	m.nop
   282da: 2f 2d 2e 1b  	l.ldi16.u r5, 0xd5d4
   282de: 01 00        	c.nop
   282e0: 00 00 00 00  	v.nop
   282e4: 00 00 00 00 00 00    	m.nop
   282ea: 6f 28 c0 00  	l.ldi_hi r5, 0x801
   282ee: 01 00        	c.nop
   282f0: 00 00 00 00  	v.nop
   282f4: 00 00 00 00 00 00    	m.nop
   282fa: 31 28 c4 00  	l.st.w r4, [r5]
   282fe: 01 00        	c.nop
   28300: 00 00 00 00  	v.nop
   28304: 00 00 00 00 00 00    	m.nop
   2830a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   2830e: 01 00        	c.nop
   28310: 00 00 00 00  	v.nop
   28314: 00 00 00 00 00 00    	m.nop
   2831a: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   2831e: 01 00        	c.nop
   28320: 00 00 00 00  	v.nop
   28324: 00 00 00 00 00 00    	m.nop
   2832a: 7b 9a 80 c0  	l.blt.s r19, r0, 0x9 <0x283b0>
   2832e: 01 00        	c.nop
   28330: 00 00 00 00  	v.nop
   28334: 00 00 00 00 00 00    	m.nop
   2833a: 00 00        	s.nop16
   2833c: 00 00        	s.nop16
   2833e: 7d 42        	<unknown>
   28340: 00 00 00 00  	v.nop
   28344: 00 00 00 00 00 00    	m.nop
   2834a: 3b 25 93 c0  	l.blt.s r4, r19, 0x14 <0x28480>
   2834e: 01 00        	c.nop
   28350: 00 00 00 00  	v.nop
   28354: 00 00 00 00 00 00    	m.nop
   2835a: 31 41 40 00  	l.ld.w r4, [r8]
   2835e: 01 00        	c.nop
   28360: 00 00 00 00  	v.nop
   28364: 00 00 00 00 00 00    	m.nop
   2836a: 63 39 93 02  	l.srl     r5, r7, r19
   2836e: 01 00        	c.nop
   28370: 00 00 00 00  	v.nop
   28374: 00 00 00 00 00 00    	m.nop
   2837a: 25 21 85 02  	l.xorix r4, r4, 0x5
   2837e: 01 00        	c.nop
   28380: 00 00 00 00  	v.nop
   28384: 00 00 00 00 00 00    	m.nop
   2838a: 31 40 c4 00  	l.st.w r4, [r8]
   2838e: 01 00        	c.nop
   28390: 00 00 00 00  	v.nop
   28394: 00 00 00 00 00 00    	m.nop
   2839a: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   2839e: 01 00        	c.nop
   283a0: 00 00 00 00  	v.nop
   283a4: 00 00 00 00 00 00    	m.nop
   283aa: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   283ae: 01 00        	c.nop
   283b0: 00 00 00 00  	v.nop
   283b4: 00 00 00 00 00 00    	m.nop
   283ba: 7b 49 80 c0  	l.blt.s r9, r0, 0x5 <0x28400>
   283be: 01 00        	c.nop
   283c0: 00 00 00 00  	v.nop
   283c4: 00 00 00 00 00 00    	m.nop
   283ca: b1 11 54 16  	l.ld.w r6, [sp, 0x174]
   283ce: 92 82        	<unknown>
   283d0: 00 00 00 00  	v.nop
   283d4: 00 00 00 00 00 00    	m.nop
   283da: 31 29 40 00  	l.ld.w r4, [r5]
   283de: 01 00        	c.nop
   283e0: 00 00 00 00  	v.nop
   283e4: 00 00 00 00 00 00    	m.nop
   283ea: 25 21 86 02  	l.xorix r4, r4, 0x6
   283ee: 01 00        	c.nop
   283f0: 00 00 00 00  	v.nop
   283f4: 00 00 00 00 00 00    	m.nop
   283fa: 31 28 c4 00  	l.st.w r4, [r5]
   283fe: 01 00        	c.nop
   28400: 00 00 00 00  	v.nop
   28404: 00 00 00 00 00 00    	m.nop
   2840a: 3b 66 80 c0  	l.blt.s r12, r0, 0x18 <0x28580>
   2840e: 01 00        	c.nop
   28410: 00 00 00 00  	v.nop
   28414: 00 00 00 00 00 00    	m.nop
   2841a: 00 00        	s.nop16
   2841c: 00 00        	s.nop16
   2841e: 7d 42        	<unknown>
   28420: 00 00 00 00  	v.nop
   28424: 00 00 00 00 00 00    	m.nop
   2842a: 3b 24 8c c0  	l.blt.s r4, r12, 0x10 <0x28520>
   2842e: 01 00        	c.nop
   28430: 00 00 00 00  	v.nop
   28434: 00 00 00 00 00 00    	m.nop
   2843a: 71 41 40 00  	l.ld.w r5, [r8]
   2843e: 01 00        	c.nop
   28440: 00 00 00 00  	v.nop
   28444: 00 00 00 00 00 00    	m.nop
   2844a: 23 39 8c 02  	l.srl     r4, r7, r12
   2844e: 01 00        	c.nop
   28450: 00 00 00 00  	v.nop
   28454: 00 00 00 00 00 00    	m.nop
   2845a: 25 29 84 02  	l.xorix r4, r5, 0x4
   2845e: 01 00        	c.nop
   28460: 00 00 00 00  	v.nop
   28464: 00 00 00 00 00 00    	m.nop
   2846a: 31 40 c4 00  	l.st.w r4, [r8]
   2846e: 01 00        	c.nop
   28470: 00 00 00 00  	v.nop
   28474: 00 00 00 00 00 00    	m.nop
   2847a: 2b 08 81 01  	l.jal r0, 0x11 <0x28580>
   2847e: 01 00        	c.nop
   28480: 00 00 00 00  	v.nop
   28484: 00 00 00 00 00 00    	m.nop
   2848a: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   2848e: 01 00        	c.nop
   28490: 00 00 00 00  	v.nop
   28494: 00 00 00 00 00 00    	m.nop
   2849a: 61 99 e0 2f  	l.addi.s r5, r19, -0x20
   2849e: 01 00        	c.nop
   284a0: 00 00 00 00  	v.nop
   284a4: 00 00 00 00 00 00    	m.nop
   284aa: 63 39 85 02  	l.srl     r5, r7, r5
   284ae: 01 00        	c.nop
   284b0: 00 00 00 00  	v.nop
   284b4: 00 00 00 00 00 00    	m.nop
   284ba: 31 31 40 00  	l.ld.w r4, [r6]
   284be: 01 00        	c.nop
   284c0: 00 00 00 00  	v.nop
   284c4: 00 00 00 00 00 00    	m.nop
   284ca: 25 21 85 02  	l.xorix r4, r4, 0x5
   284ce: 01 00        	c.nop
   284d0: 00 00 00 00  	v.nop
   284d4: 00 00 00 00 00 00    	m.nop
   284da: 31 30 c4 00  	l.st.w r4, [r6]
   284de: 01 00        	c.nop
   284e0: 00 00 00 00  	v.nop
   284e4: 00 00 00 00 00 00    	m.nop
   284ea: 2f 25 2e 1b  	l.ldi16.u r4, 0xd5d4
   284ee: 01 00        	c.nop
   284f0: 00 00 00 00  	v.nop
   284f4: 00 00 00 00 00 00    	m.nop
   284fa: 6f 20 c0 00  	l.ldi_hi r4, 0x801
   284fe: 01 00        	c.nop
   28500: 00 00 00 00  	v.nop
   28504: 00 00 00 00 00 00    	m.nop
   2850a: 3b 4b e0 bf  	l.bge.s r9, r0, -0x14 <0x283c0>
   2850e: 01 00        	c.nop
   28510: 00 00 00 00  	v.nop
   28514: 00 00 00 00 00 00    	m.nop
   2851a: 2b 78 fe 3f  	l.jal r0, -0x11 <0x28400>
   2851e: 01 00        	c.nop
   28520: 00 00 00 00  	v.nop
   28524: 00 00 00 00 00 00    	m.nop
   2852a: b1 11 44 02  	l.ld.w r6, [sp, 0x24]
   2852e: 01 00        	c.nop
   28530: 00 00 00 00  	v.nop
   28534: 00 00 00 00 00 00    	m.nop
   2853a: 21 61 e0 2f  	l.addi.s r4, r12, -0x20
   2853e: 01 00        	c.nop
   28540: 00 00 00 00  	v.nop
   28544: 00 00 00 00 00 00    	m.nop
   2854a: 23 39 84 02  	l.srl     r4, r7, r4
   2854e: 01 00        	c.nop
   28550: 00 00 00 00  	v.nop
   28554: 00 00 00 00 00 00    	m.nop
   2855a: 71 31 40 00  	l.ld.w r5, [r6]
   2855e: 01 00        	c.nop
   28560: 00 00 00 00  	v.nop
   28564: 00 00 00 00 00 00    	m.nop
   2856a: 25 29 84 02  	l.xorix r4, r5, 0x4
   2856e: 01 00        	c.nop
   28570: 00 00 00 00  	v.nop
   28574: 00 00 00 00 00 00    	m.nop
   2857a: 31 30 c4 00  	l.st.w r4, [r6]
   2857e: 01 00        	c.nop
   28580: 00 00 00 00  	v.nop
   28584: 00 00 00 00 00 00    	m.nop
   2858a: 2f 27 6d 21  	l.ldi16.s r4, 0xdbc
   2858e: 01 00        	c.nop
   28590: 00 00 00 00  	v.nop
   28594: 00 00 00 00 00 00    	m.nop
   2859a: 2f 40 00 21  	l.ldi16.s r8, 0x0
   2859e: 01 00        	c.nop
   285a0: 00 00 00 00  	v.nop
   285a4: 00 00 00 00 00 00    	m.nop
   285aa: 23 11 04 18  	<unknown>
   285ae: 1b 84        	<unknown>
   285b0: 00 00 00 00  	v.nop
   285b4: 00 00 00 00 00 00    	m.nop
   285ba: f1 20 40 00  	l.ld.w lr, [r4]
   285be: 17 84        	<unknown>
   285c0: 00 00 00 00  	v.nop
   285c4: 00 00 00 00 00 00    	m.nop
   285ca: 00 00        	s.nop16
   285cc: 00 00        	s.nop16
   285ce: 1f 84        	<unknown>
   285d0: 00 00 00 00  	v.nop
   285d4: 00 00 00 00 00 00    	m.nop
   285da: 00 00        	s.nop16
   285dc: 00 00        	s.nop16
   285de: 03 8c        	<unknown>
   285e0: 00 00 00 00  	v.nop
   285e4: 00 00 00 00 00 00    	m.nop
   285ea: 00 00        	s.nop16
   285ec: 00 00        	s.nop16
   285ee: 23 84        	<unknown>
   285f0: 00 00 00 00  	v.nop
   285f4: 00 00 00 00 00 00    	m.nop
   285fa: 00 00        	s.nop16
   285fc: 00 00        	s.nop16
   285fe: 7e 81        	<unknown>
   28600: 00 00 00 00  	v.nop
   28604: 00 00 00 00 00 00    	m.nop
   2860a: f1 17 58 fe  	l.ld.w r31, [sp, -0x8]
   2860e: 01 00        	c.nop
   28610: 00 00 00 00  	v.nop
   28614: 00 00 00 00 00 00    	m.nop
   2861a: f1 10 5c fe  	l.ld.w lr, [sp, -0x4]
   2861e: 01 00        	c.nop
   28620: 00 00 00 00  	v.nop
   28624: 00 00 00 00 00 00    	m.nop
   2862a: 2b 18 00 01  	l.jalr r0, lr, 0x0
   2862e: 01 00        	c.nop
   28630: 00 00 00 00  	v.nop
   28634: 00 00 00 00 00 00    	m.nop
   2863a: 2f 20 00 01  	l.ldi16.u r4, 0x0
   2863e: 81 43        	<unknown>
   28640: 00 00 00 00  	v.nop
   28644: 00 00 00 00 00 00    	m.nop
   2864a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   2864e: 01 00        	c.nop
   28650: 00 00 00 00  	v.nop
   28654: 00 00 00 00 00 00    	m.nop
   2865a: 31 11 c4 24  	l.st.w r4, [sp, 0x244]
   2865e: 01 00        	c.nop
   28660: 00 00 00 00  	v.nop
   28664: 00 00 00 00 00 00    	m.nop
   2866a: 2f 20 00 01  	l.ldi16.u r4, 0x0
   2866e: 01 00        	c.nop
   28670: 00 00 00 00  	v.nop
   28674: 00 00 00 00 00 00    	m.nop
   2867a: 2f 20 80 00  	l.ldi_hi r4, 0x0
   2867e: 01 00        	c.nop
   28680: 00 00 00 00  	v.nop
   28684: 00 00 00 00 00 00    	m.nop
   2868a: 31 10 c4 24  	l.st.w r4, [sp, 0x240]
   2868e: 01 00        	c.nop
   28690: 00 00 00 00  	v.nop
   28694: 00 00 00 00 00 00    	m.nop
   2869a: 2f 20 00 01  	l.ldi16.u r4, 0x0
   2869e: 01 00        	c.nop
   286a0: 00 00 00 00  	v.nop
   286a4: 00 00 00 00 00 00    	m.nop
   286aa: af 31 05 21  	l.ldi16.s r6, 0xa6
   286ae: 01 00        	c.nop
   286b0: 00 00 00 00  	v.nop
   286b4: 00 00 00 00 00 00    	m.nop
   286ba: 2f 20 80 00  	l.ldi_hi r4, 0x0
   286be: 01 00        	c.nop
   286c0: 00 00 00 00  	v.nop
   286c4: 00 00 00 00 00 00    	m.nop
   286ca: 2f 28 00 01  	l.ldi16.u r5, 0x0
   286ce: 01 00        	c.nop
   286d0: 00 00 00 00  	v.nop
   286d4: 00 00 00 00 00 00    	m.nop
   286da: 31 12 c6 24  	l.st.w r6, [sp, 0x248]
   286de: 01 00        	c.nop
   286e0: 00 00 00 00  	v.nop
   286e4: 00 00 00 00 00 00    	m.nop
   286ea: 31 14 c4 24  	l.st.w r4, [sp, 0x250]
   286ee: 01 00        	c.nop
   286f0: 00 00 00 00  	v.nop
   286f4: 00 00 00 00 00 00    	m.nop
   286fa: 2f 20 60 1b  	l.ldi16.u r4, 0xdc00
   286fe: 01 00        	c.nop
   28700: 00 00 00 00  	v.nop
   28704: 00 00 00 00 00 00    	m.nop
   2870a: 2f 28 80 00  	l.ldi_hi r5, 0x0
   2870e: 01 00        	c.nop
   28710: 00 00 00 00  	v.nop
   28714: 00 00 00 00 00 00    	m.nop
   2871a: 6f 22 80 00  	l.ldi_hi r4, 0x9
   2871e: 01 00        	c.nop
   28720: 00 00 00 00  	v.nop
   28724: 00 00 00 00 00 00    	m.nop
   2872a: 31 13 c5 24  	l.st.w r5, [sp, 0x24c]
   2872e: 01 00        	c.nop
   28730: 00 00 00 00  	v.nop
   28734: 00 00 00 00 00 00    	m.nop
   2873a: 31 15 c4 24  	l.st.w r4, [sp, 0x254]
   2873e: 01 00        	c.nop
   28740: 00 00 00 00  	v.nop
   28744: 00 00 00 00 00 00    	m.nop
   2874a: 31 11 5c 02  	l.ld.w r4, [sp, 0x3c]
   2874e: 01 00        	c.nop
   28750: 00 00 00 00  	v.nop
   28754: 00 00 00 00 00 00    	m.nop
   2875a: 2f 28 00 01  	l.ldi16.u r5, 0x0
   2875e: 01 00        	c.nop
   28760: 00 00 00 00  	v.nop
   28764: 00 00 00 00 00 00    	m.nop
   2876a: 2f 40 13 07  	l.ldi16.u r8, 0x3260
   2876e: 01 00        	c.nop
   28770: 00 00 00 00  	v.nop
   28774: 00 00 00 00 00 00    	m.nop
   2877a: 2f 28 80 00  	l.ldi_hi r5, 0x0
   2877e: 01 00        	c.nop
   28780: 00 00 00 00  	v.nop
   28784: 00 00 00 00 00 00    	m.nop
   2878a: a1 11 40 22  	l.addi.s r6, sp, 0x240
   2878e: 01 00        	c.nop
   28790: 00 00 00 00  	v.nop
   28794: 00 00 00 00 00 00    	m.nop
   2879a: 31 16 c4 24  	l.st.w r4, [sp, 0x258]
   2879e: 01 00        	c.nop
   287a0: 00 00 00 00  	v.nop
   287a4: 00 00 00 00 00 00    	m.nop
   287aa: 2f 20 00 01  	l.ldi16.u r4, 0x0
   287ae: 01 00        	c.nop
   287b0: 00 00 00 00  	v.nop
   287b4: 00 00 00 00 00 00    	m.nop
   287ba: 2f 20 80 00  	l.ldi_hi r4, 0x0
   287be: 01 00        	c.nop
   287c0: 00 00 00 00  	v.nop
   287c4: 00 00 00 00 00 00    	m.nop
   287ca: ef 41 80 00  	l.ldi_hi r8, 0x7
   287ce: 01 00        	c.nop
   287d0: 00 00 00 00  	v.nop
   287d4: 00 00 00 00 00 00    	m.nop
   287da: eb 40 00 01  	l.jalr lr, r8, 0x0
   287de: 01 00        	c.nop
   287e0: 00 00 00 00  	v.nop
   287e4: 00 00 00 00 00 00    	m.nop
   287ea: 2f 20 7c 1f  	l.ldi16.u r4, 0xff80
   287ee: 01 00        	c.nop
   287f0: 00 00 00 00  	v.nop
   287f4: 00 00 00 00 00 00    	m.nop
   287fa: af 21 80 00  	l.ldi_hi r4, 0x6
   287fe: 01 00        	c.nop
   28800: 00 00 00 00  	v.nop
   28804: 00 00 00 00 00 00    	m.nop
   2880a: eb 20 00 01  	l.jalr lr, r4, 0x0
   2880e: 01 00        	c.nop

