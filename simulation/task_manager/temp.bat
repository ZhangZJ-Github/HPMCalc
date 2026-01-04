cd /d F:\changeworld\PyDISK\_dirty_works\HighQualityEBeamGeneration2\GPT_rundir\backup\swlinac.template_20260101022909_808136192.in
rem gpt -v -o swlinac.gdf swlinac.in  > 1.log 2>&1
rem start excel
copy swlinac.gdf swlinac_copy.gdf
gdftrans -o traj.gdf swlinac.gdf time x y z G nmacro q Bz fBz
gdfa -v -o std.gdf swlinac.gdf position stdx stdy stdt avgG nemixrms nemiyrms nemix90 nemiy90 Q stdG
gdf2his -b -n nmacro -o hist.gdf swlinac.gdf G 0.1
rem code 1.log