cd /d F:\changeworld\PyDISK\_dirty_works\SW_linac_design\GPT_rundir
gpt -v -o swlinac.gdf swlinac.in  > 1.log 2>&1
gdftrans -o traj.gdf swlinac.gdf time x y z G nmacro q Bz fBz
gdfa -v -o std.gdf swlinac.gdf position stdx stdy stdt avgG nemixrms nemiyrms Q stdG
gdf2his -b -n nmacro -o hist.gdf swlinac.gdf G 0.1
rem code 1.log