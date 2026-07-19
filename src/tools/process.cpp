#include "agents_framework/tools/process.hpp"

#include <algorithm>
#include <thread>
#include <utility>

#ifdef _WIN32

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

namespace agents_framework::tools {

namespace {

std::wstring widen(const std::string& text) {
  if (text.empty()) return {};
  const int size =
      MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
  std::wstring wide(static_cast<std::size_t>(size), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(), size);
  return wide;
}

void append_quoted(std::wstring& line, const std::wstring& arg) {
  if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) {
    line += arg;
    return;
  }
  line += L'"';
  std::size_t backslashes = 0;
  for (const wchar_t ch : arg) {
    if (ch == L'\\') {
      ++backslashes;
      continue;
    }
    if (ch == L'"') {
      line.append(backslashes * 2 + 1, L'\\');
      line += L'"';
    } else {
      line.append(backslashes, L'\\');
      line += ch;
    }
    backslashes = 0;
  }
  line.append(backslashes * 2, L'\\');
  line += L'"';
}

}

core::Result<ProcessResult> run_process(const std::vector<std::string>& argv,
                                        const ProcessOptions& options) {
  if (argv.empty()) {
    return core::fail(core::ErrorCode::Invalid, "process command must not be empty");
  }

  SECURITY_ATTRIBUTES inheritable{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE out_read = nullptr;
  HANDLE out_write = nullptr;
  HANDLE in_read = nullptr;
  HANDLE in_write = nullptr;
  if (!CreatePipe(&out_read, &out_write, &inheritable, 0)) {
    return core::fail(core::ErrorCode::Io, "failed to create output pipe");
  }
  SetHandleInformation(out_read, HANDLE_FLAG_INHERIT, 0);
  if (!CreatePipe(&in_read, &in_write, &inheritable, 0)) {
    CloseHandle(out_read);
    CloseHandle(out_write);
    return core::fail(core::ErrorCode::Io, "failed to create input pipe");
  }
  SetHandleInformation(in_write, HANDLE_FLAG_INHERIT, 0);

  std::wstring command_line;
  for (const std::string& arg : argv) {
    if (!command_line.empty()) command_line += L' ';
    append_quoted(command_line, widen(arg));
  }

  HANDLE inherit_list[2] = {in_read, out_write};
  SIZE_T attr_size = 0;
  InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
  std::vector<unsigned char> attr_storage(attr_size);
  auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_storage.data());
  if (!InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size) ||
      !UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherit_list,
                                 sizeof(inherit_list), nullptr, nullptr)) {
    CloseHandle(out_read);
    CloseHandle(out_write);
    CloseHandle(in_read);
    CloseHandle(in_write);
    return core::fail(core::ErrorCode::Io, "failed to prepare process attributes");
  }

  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = in_read;
  startup.StartupInfo.hStdOutput = out_write;
  startup.StartupInfo.hStdError = out_write;
  startup.lpAttributeList = attrs;

  HANDLE job = CreateJobObjectW(nullptr, nullptr);
  if (job != nullptr) {
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits));
  }

  PROCESS_INFORMATION process{};
  const BOOL started = CreateProcessW(
      nullptr, command_line.data(), nullptr, nullptr, TRUE,
      CREATE_NO_WINDOW | CREATE_SUSPENDED | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
      &startup.StartupInfo, &process);
  DeleteProcThreadAttributeList(attrs);
  CloseHandle(out_write);
  CloseHandle(in_read);
  if (!started) {
    if (job != nullptr) CloseHandle(job);
    CloseHandle(out_read);
    CloseHandle(in_write);
    return core::fail(core::ErrorCode::Io, "failed to start process", argv.front());
  }
  const bool in_job = job != nullptr && AssignProcessToJobObject(job, process.hProcess) != 0;
  ResumeThread(process.hThread);

  std::thread writer([&] {
    const std::string& data = options.stdin_data;
    std::size_t offset = 0;
    while (offset < data.size()) {
      DWORD written = 0;
      if (!WriteFile(in_write, data.data() + offset,
                     static_cast<DWORD>(std::min<std::size_t>(data.size() - offset, 1 << 16)),
                     &written, nullptr) ||
          written == 0) {
        break;
      }
      offset += written;
    }
    CloseHandle(in_write);
  });

  ProcessResult result;
  std::thread reader([&] {
    char buffer[4096];
    DWORD read = 0;
    while (ReadFile(out_read, buffer, sizeof(buffer), &read, nullptr) && read > 0) {
      if (result.output.size() < options.max_output_bytes) {
        const std::size_t room = options.max_output_bytes - result.output.size();
        result.output.append(buffer, std::min<std::size_t>(read, room));
        if (read > room) result.truncated = true;
      } else {
        result.truncated = true;
      }
    }
  });

  const DWORD wait_ms = options.timeout.count() > 0
                            ? static_cast<DWORD>(options.timeout.count())
                            : INFINITE;
  const bool timed_out = WaitForSingleObject(process.hProcess, wait_ms) == WAIT_TIMEOUT;
  if (timed_out) {
    if (in_job) {
      TerminateJobObject(job, 1);
    } else {
      TerminateProcess(process.hProcess, 1);
    }
  }
 
  if (job != nullptr) CloseHandle(job);

  reader.join();
  writer.join();

  DWORD exit_code = 0;
  GetExitCodeProcess(process.hProcess, &exit_code);
  CloseHandle(process.hProcess);
  CloseHandle(process.hThread);
  CloseHandle(out_read);

  if (timed_out) {
    return core::fail(core::ErrorCode::Timeout, "process timed out", argv.front());
  }
  result.exit_code = static_cast<int>(exit_code);
  return result;
}

}

#else

#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace agents_framework::tools {

core::Result<ProcessResult> run_process(const std::vector<std::string>& argv,
                                        const ProcessOptions& options) {
  if (argv.empty()) {
    return core::fail(core::ErrorCode::Invalid, "process command must not be empty");
  }

  int out_pipe[2];
  int in_pipe[2];
  if (pipe2(out_pipe, O_CLOEXEC) != 0) {
    return core::fail(core::ErrorCode::Io, "failed to create output pipe");
  }
  if (pipe2(in_pipe, O_CLOEXEC) != 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    return core::fail(core::ErrorCode::Io, "failed to create input pipe");
  }

  std::vector<std::string> args = argv;
  std::vector<char*> argv_c;
  argv_c.reserve(args.size() + 1);
  for (std::string& arg : args) argv_c.push_back(arg.data());
  argv_c.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    close(out_pipe[0]);
    close(out_pipe[1]);
    close(in_pipe[0]);
    close(in_pipe[1]);
    return core::fail(core::ErrorCode::Io, "failed to fork", argv.front());
  }
  if (pid == 0) {
    setpgid(0, 0);
    dup2(in_pipe[0], STDIN_FILENO);
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(out_pipe[1], STDERR_FILENO);
    execvp(argv_c[0], argv_c.data());
    _exit(127);
  }
  setpgid(pid, pid);
  close(out_pipe[1]);
  close(in_pipe[0]);

  std::thread writer([&] {
    sigset_t sigpipe;
    sigemptyset(&sigpipe);
    sigaddset(&sigpipe, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &sigpipe, nullptr);
    const std::string& data = options.stdin_data;
    std::size_t offset = 0;
    while (offset < data.size()) {
      const ssize_t written = write(in_pipe[1], data.data() + offset, data.size() - offset);
      if (written <= 0) {
        if (written < 0 && errno == EINTR) continue;
        break;
      }
      offset += static_cast<std::size_t>(written);
    }
    close(in_pipe[1]);
  });

  const auto deadline = std::chrono::steady_clock::now() + options.timeout;
  ProcessResult result;
  bool timed_out = false;
  char buffer[4096];
  for (;;) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (options.timeout.count() > 0 && remaining.count() <= 0) {
      timed_out = true;
      break;
    }
    pollfd waiting{out_pipe[0], POLLIN, 0};
    const int ready = poll(&waiting, 1,
                           options.timeout.count() > 0 ? static_cast<int>(remaining.count()) : -1);
    if (ready == 0) {
      timed_out = true;
      break;
    }
    if (ready < 0) {
      if (errno == EINTR) continue;
      break;
    }
    const ssize_t read_count = read(out_pipe[0], buffer, sizeof(buffer));
    if (read_count < 0 && errno == EINTR) continue;
    if (read_count <= 0) break;
    const auto count = static_cast<std::size_t>(read_count);
    if (result.output.size() < options.max_output_bytes) {
      const std::size_t room = options.max_output_bytes - result.output.size();
      result.output.append(buffer, std::min(count, room));
      if (count > room) result.truncated = true;
    } else {
      result.truncated = true;
    }
  }

  if (timed_out) {
    if (kill(-pid, SIGKILL) != 0) kill(pid, SIGKILL);
  }
  writer.join();
  close(out_pipe[0]);

  int status = 0;
  for (;;) {
    const pid_t reaped = waitpid(pid, &status, WNOHANG);
    if (reaped == pid) break;
    if (reaped < 0) break;
    if (std::chrono::steady_clock::now() >= deadline && options.timeout.count() > 0) {
      if (kill(-pid, SIGKILL) != 0) kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      timed_out = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  if (timed_out) {
    return core::fail(core::ErrorCode::Timeout, "process timed out", argv.front());
  }
  if (WIFEXITED(status)) {
    result.exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    result.exit_code = 128 + WTERMSIG(status);
  } else {
    result.exit_code = -1;
  }
  return result;
}

}

#endif
