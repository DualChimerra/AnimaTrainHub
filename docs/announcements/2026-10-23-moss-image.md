---
date: 2026-10-23
tag: notice
title: MOSS-Image training and shared model lists
---

The development branch adds MOSS-Image Base, independent ConvRot INT8 options
for transformer and text encoder weights, caption caching and layer offloading.
Settings now lists Checkpoints, VAE and Text Encoders discovered in selected
folders. Existing family defaults and shared training controls are retained.

CPU and checkpoint-header checks are available; full-size GPU validation is
pending. See the MOSS user guide for supported formats, memory controls and rollback.
