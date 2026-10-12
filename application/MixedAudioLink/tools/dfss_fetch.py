#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dfss_fetch.py —— 直连 Sophgo 官方 SFTP 源的下载工具（dfss 的轻量替代实现）

背景:
    官方 dfss 客户端（pip install dfss）内部带"测速选服务器"环节，在部分网络下
    会一直返回 speed 0.00 Byte/s 从而导致 "No available server found"。
    但底层 SFTP 服务本身是通的，本脚本用 paramiko 直接连接即可正常下载。

用法:
    python dfss_fetch.py ls  <远端目录>                 # 列出目录（含大小）
    python dfss_fetch.py get <远端文件> [本地路径]       # 下载文件（带进度）
    python dfss_fetch.py hosts                          # 测试各服务器可用性

远端路径写法与官方 dfss 一致，例如:
    python dfss_fetch.py get /sophon-stream/docker/stream_dev.tar

依赖: paramiko  (pip install paramiko)
"""

import json
import os
import posixpath
import sys
import time

import paramiko

# Sophgo 官方 dfss SFTP 服务器列表（取自 dfss 包内的 dfss-config.json）
# 实测 106.38.208.114 更稳定，故排在前面优先尝试
SERVERS = [
    ("106.38.208.114", 32022, "open", "open"),
    ("103.68.183.114", 32022, "open", "open"),
]
CONNECT_TIMEOUT = 20
AUTH_TIMEOUT = 25


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return "%.2f %s" % (n, unit)
        n /= 1024.0
    return "%.2f PB" % n


def _try_connect(host, port, user, pwd):
    sock = __import__("socket").create_connection((host, port), timeout=CONNECT_TIMEOUT)
    tr = paramiko.Transport(sock)
    tr.banner_timeout = 30
    tr.auth_timeout = AUTH_TIMEOUT
    tr.start_client(timeout=CONNECT_TIMEOUT)
    tr.set_keepalive(15)
    tr.auth_password(user, pwd)
    sftp = paramiko.SFTPClient.from_transport(tr)
    return sftp, tr, sock


def connect(retries=3):
    """依次尝试各服务器（含重试），返回 (sftp, transport, sock, host)。"""
    last = None
    for attempt in range(1, retries + 1):
        for host, port, user, pwd in SERVERS:
            try:
                sftp, tr, sock = _try_connect(host, port, user, pwd)
                print("[dfss_fetch] 已连接 %s:%d (user=%s)" % (host, port, user))
                return sftp, tr, sock, host
            except Exception as e:
                print("[dfss_fetch] %s 连接失败: %s: %s" % (host, type(e).__name__, e))
                last = e
        if attempt < retries:
            print("[dfss_fetch] 第 %d 轮全部失败，3 秒后重试..." % attempt)
            time.sleep(3)
    raise SystemExit("所有服务器均不可用: %s" % last)


def cmd_hosts():
    for host, port, user, pwd in SERVERS:
        try:
            s = __import__("socket").create_connection((host, port), timeout=CONNECT_TIMEOUT)
            banner = s.recv(64).decode("utf-8", "ignore").strip()
            s.close()
            print("OK   %s:%d  %s" % (host, port, banner))
        except Exception as e:
            print("FAIL %s:%d  %s" % (host, port, e))


def cmd_ls(remote):
    sftp, tr, sock, _ = connect()
    try:
        entries = sorted(sftp.listdir_attr(remote), key=lambda a: a.filename)
        print("\n%s  (%d 项)" % (remote, len(entries)))
        print("%-52s %12s  %s" % ("名称", "大小", "类型"))
        print("-" * 78)
        total = 0
        for a in entries:
            isdir = (a.st_mode or 0) & 0o040000
            size = a.st_size or 0
            if not isdir:
                total += size
            print("%-52s %12s  %s" % (a.filename, "-" if isdir else human(size),
                                      "dir" if isdir else "file"))
        print("-" * 78)
        print("文件合计: %s" % human(total))
    finally:
        sftp.close()
        tr.close()
        sock.close()


def cmd_get(remote, local=None):
    if local is None:
        local = os.path.basename(remote.rstrip("/")) or "download.bin"
    if os.path.isdir(local):
        local = os.path.join(local, os.path.basename(remote.rstrip("/")))
    local = os.path.abspath(local)

    sftp, tr, sock, host = connect()
    try:
        st = sftp.stat(remote)
        total = st.st_size
        print("[dfss_fetch] 远端 %s (%s)" % (remote, human(total)))
        print("[dfss_fetch] 本地 %s" % local)

        # 断点续传
        done = os.path.getsize(local) if os.path.exists(local) else 0
        if done > total:
            done = 0
        mode = "ab" if done else "wb"
        if done:
            print("[dfss_fetch] 检测到已存在 %s，从该位置续传" % human(done))

        t0 = time.time()
        last = [0.0]

        def cb(transferred, tot):
            now = time.time()
            if now - last[0] < 2.0 and transferred != tot:
                return
            last[0] = now
            got = done + transferred
            pct = got * 100.0 / tot if tot else 0
            speed = transferred / max(now - t0, 1e-6)
            bar_len = 28
            filled = int(bar_len * pct / 100)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write("\r  [%s] %6.2f%%  %s / %s  %s/s   " %
                             (bar, pct, human(got), human(tot), human(speed)))
            sys.stdout.flush()

        with sftp.open(remote, "rb") as rf:
            if done:
                rf.seek(done)
            with open(local, mode) as lf:
                rf.prefetch(total - done)
                transferred = 0
                while transferred < total - done:
                    chunk = rf.read(min(1024 * 1024, total - done - transferred))
                    if not chunk:
                        break
                    lf.write(chunk)
                    transferred += len(chunk)
                    cb(transferred, total)
        cb(total - done, total)
        print("\n[dfss_fetch] 完成，耗时 %.1fs" % (time.time() - t0))
    finally:
        sftp.close()
        tr.close()
        sock.close()
    return local


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    action = sys.argv[1].lower()
    if action == "hosts":
        cmd_hosts()
    elif action == "ls":
        if len(sys.argv) < 3:
            print("用法: python dfss_fetch.py ls <远端目录>")
            return 2
        cmd_ls(sys.argv[2])
    elif action == "get":
        if len(sys.argv) < 3:
            print("用法: python dfss_fetch.py get <远端文件> [本地路径]")
            return 2
        cmd_get(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
