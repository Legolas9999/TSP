import subprocess

# 定义命令和输出文件,记得cd到相应目录
for i in range(5, 201):
    command = f"./concorde -o /home/chongjiu/concorde_test/solution/random{i}.sol /home/chongjiu/concorde_test/mat/random{i}.tsp"
    output_log = f"/home/chongjiu/concorde_test/time_log/random{i}_output.log"

    # 执行命令并捕获输出
    try:
        # 使用 subprocess 运行命令
        result = subprocess.run(
            command,
            shell=True,               # 使用 shell 解释命令
            stdout=subprocess.PIPE,   # 捕获标准输出
            stderr=subprocess.PIPE,   # 捕获标准错误
            text=True                 # 输出以字符串形式返回
        )

        # 将命令输出写入日志文件
        with open(output_log, "w") as log_file:
            log_file.write("Standard Output:\n")
            log_file.write(result.stdout)  # 写入标准输出
            log_file.write("\nStandard Error:\n")
            log_file.write(result.stderr)  # 写入标准错误

        print(f"命令执行完成，日志已保存到: {output_log}")

    except Exception as e:
        print(f"命令执行失败: {e}")


