# PLAN-A Drug Utilization Review (DUR) Policy
Document version: 2026.02 | Effective: 2026-01-01

## Purpose
Concurrent DUR screens every claim at adjudication for safety concerns. When a
clinically significant conflict is detected, the claim is rejected with NCPDP
reject code 88 and a DUR conflict code is returned.

## Conflict Types
- DD (Drug-Drug Interaction): significant interaction with an active medication
  in the member's profile
- TD (Therapeutic Duplication): duplicate therapy within the same class
- HD (High Dose): dose exceeds the maximum recommended daily dose
- LD (Low Dose): dose below the recognised therapeutic minimum
- MC (Drug-Disease Contraindication): conflicts with a documented condition
- PG (Pregnancy Alert)

## Resolving a DUR Reject
A DUR reject requires pharmacist review before the claim can be resubmitted.
It is NOT resolved by a prior authorization and cannot be overridden by the
plan help desk.

The reviewing pharmacist evaluates the conflict and, if the therapy is
clinically appropriate, resubmits the claim with the appropriate NCPDP DUR
override codes: Reason for Service, Professional Service, and Result of Service.

If the conflict is NOT clinically appropriate, the pharmacist must contact the
prescriber before dispensing. High Dose and Drug-Drug conflicts of severity
level 1 always require prescriber contact and may not be overridden by the
pharmacist alone.

Every override must be documented in the pharmacy record with the reviewing
pharmacist's identifier and the clinical rationale.
