Timeline Manager Documentation
==============================

Overview
--------

The Job Scheduler is a Python library designed for scheduling and managing recurring tasks (jobs). This library allows you to easily set up jobs that can run at specific intervals (e.g., every minute, every hour, etc.), on specific days, or until a certain time, with support for repetition. Threading is also supported and optional for non-blocking execution.

Features
--------

- **Flexible Scheduling**: Schedule jobs to run every few seconds, minutes, hours, days, weeks, months, or even years.
- **Specific Time Execution**: Run jobs at a specific time of day or on specific days of the week or month.
- **Repeat and Until**: Set jobs to run a certain number of times or until a specific date and time.
- **Threading Support**: Optionally execute jobs in separate threads for non-blocking execution.
- **Two Scheduling Methods**: Jobs can be scheduled using a decorator-based approach or a standard object-oriented method.
- **Simultaneous Job Execution**: Multiple jobs can be scheduled to run at the same time.
- **Easy Job Management**: Add, execute, and manage multiple jobs with a simple API.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Usage

