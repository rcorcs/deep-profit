  %retval = alloca i32
  %argc.addr = alloca i32
  %argv.addr = alloca i8**
  store i32 0, i32* %retval
  store i32 %argc, i32* %argc.addr
  store i8** %argv, i8*** %argv.addr
  %0 = load i32, i32* %argc.addr
  %call = call i32 @square(i32 %0)
  %call1 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([4 x i8], [4 x i8]* @.str, i64 0, i64 0), i32 %call)
  ret i32 0
