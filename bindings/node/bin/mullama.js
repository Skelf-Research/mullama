#!/usr/bin/env node
/* eslint-disable */

'use strict'

const { spawnSync, execSync } = require('child_process')
const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

function isMusl() {
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      const lddPath = execSync('which ldd').toString().trim()
      return readFileSync(lddPath, 'utf8').includes('musl')
    } catch (_) {
      return true
    }
  }
  const { glibcVersionRuntime } = process.report.getReport().header
  return !glibcVersionRuntime
}

function triple() {
  switch (platform) {
    case 'win32':
      if (arch === 'x64') return 'win32-x64-msvc'
      if (arch === 'arm64') return 'win32-arm64-msvc'
      if (arch === 'ia32') return 'win32-ia32-msvc'
      break
    case 'darwin':
      if (arch === 'x64') return 'darwin-x64'
      if (arch === 'arm64') return 'darwin-arm64'
      break
    case 'linux':
      if (arch === 'x64') return isMusl() ? 'linux-x64-musl' : 'linux-x64-gnu'
      if (arch === 'arm64') return isMusl() ? 'linux-arm64-musl' : 'linux-arm64-gnu'
      if (arch === 'arm') return isMusl() ? 'linux-arm-musleabihf' : 'linux-arm-gnueabihf'
      break
    case 'freebsd':
      if (arch === 'x64') return 'freebsd-x64'
      break
  }
  return null
}

function resolveBinary() {
  const t = triple()
  if (!t) return null
  const exeSuffix = platform === 'win32' ? '.exe' : ''
  const pkgRoot = join(__dirname, '..')
  const candidates = [
    join(pkgRoot, `mullama-cli.${t}${exeSuffix}`),
    join(pkgRoot, 'npm', t, `mullama${exeSuffix}`),
    join(pkgRoot, 'bin', `mullama-cli.${t}${exeSuffix}`),
  ]
  for (const p of candidates) {
    if (existsSync(p)) return p
  }
  return null
}

const binary = resolveBinary()
if (!binary) {
  process.stderr.write(
    `mullama: no bundled binary found for ${platform}-${arch}. ` +
      `Download a release from https://github.com/cognisoc/mullama/releases\n`,
  )
  process.exit(1)
}

const result = spawnSync(binary, process.argv.slice(2), { stdio: 'inherit' })
if (result.error) {
  process.stderr.write(`mullama: failed to spawn ${binary}: ${result.error.message}\n`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)
