gpt -v -o test_acc.gdf test_acc.in  > 1.log 2>&1 
gdftrans -o traj.gdf test_acc.gdf time x y z G nmacro q Bz
rem gdfa -v -o std.gdf test_acc.gdf position stdx stdy avgG nemixrms nemiyrms Q stdG
rem gdf2his -b -n nmacro -o hist.gdf test_acc.gdf G 0.1
rem code 1.log