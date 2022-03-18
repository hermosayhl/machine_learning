file(REMOVE_RECURSE
  "../bin/PROJECT_NAME.exe"
  "../bin/PROJECT_NAME.exe.manifest"
  "../bin/PROJECT_NAME.pdb"
  "CMakeFiles/PROJECT_NAME.dir/src/svm.cpp.obj"
  "libPROJECT_NAME.dll.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/PROJECT_NAME.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
