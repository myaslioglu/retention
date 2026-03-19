# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Storage & External Devices

### GitHub

- **Username:** myaslioglu
- **Repo:** https://github.com/myaslioglu/retention (private, sonra public yapılacak)
- **Auth:** GitHub CLI (`gh`) ile giriş yapıldı
- **Token scopes:** repo, workflow, gist, notifications, user, write:packages (full access)
- **Note:** Retention → HaciCognitiveNet push edildi (16 Mart 2026)
- **Config:** `git config user.email "openclaw@haci.ai"` / `user.name "Hacı (OpenClaw)"`

## External Hard Disk
- **Name:** Murat Hardisk
- **Mount Path:** `/Volumes/Murat`
- **Connected:** Yes (2026-03-03)
- **Purpose:** Extra storage for retention module data, large datasets, model checkpoints, OpenClaw backups
- **Usage:** When retention module needs more space, use this external drive
- **Backup Command:** `rm -rf /Volumes/Murat/openclawbackup 2>/dev/null; cp -r ~/.openclaw /Volumes/Murat/openclawbackup`
- **Note:** Overwrite on each backup (Başkan's instruction)

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

Add whatever helps you do your job. This is your cheat sheet.
