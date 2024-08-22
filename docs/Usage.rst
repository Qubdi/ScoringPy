
Usage
-----

Scheduling Methods
##################

The Job Scheduler provides two main methods for scheduling tasks: the **Standard Method** and the **Decorator Method**. Below are examples of how to use both methods.

Basic Example (Standard Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define a standard task function
   def my_task():
       print("Task executed!")

   # Create a job using the standard method
   job = Job().every(1).minute.repeat(3).do(my_task)

   # Add the job to the scheduler
   scheduler.add_job(job)

   # Start the scheduler
   scheduler.run_pending()

Basic Example (Decorator Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define a sample job function using the decorator method
   @jobs(interval=1, unit='minute', repeat=3, scheduler=scheduler)
   def my_task():
       print("Task executed!")

   # Start the scheduler
   scheduler.run_pending()

In both examples:

- The job is scheduled to run every minute.
- It will repeat 3 times before stopping.

Scheduling Options
##################

The `Job` class provides a variety of options for scheduling jobs. These options can be chained together to create complex scheduling scenarios.

Setting Interval and Unit
~~~~~~~~~~~~~~~~~~~~~~~~~

The `interval` and `unit` methods define how often the job should run. Supported units include:

- `second`
- `minute`
- `hour`
- `day`
- `week`
- `month`
- `year`

.. code-block:: python

   job.every(5).minute.do(my_task)

This will schedule the job to run every 5 minutes.

Examples with `every()` for Intervals Greater Than 1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running a Job Every 2 Hours
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define the task function
   def task_every_2_hours():
       print("Task executed every 2 hours!")

   # Create a job to run every 2 hours
   job = Job().every(2).hour.do(task_every_2_hours)

   # Add the job to the scheduler
   scheduler.add_job(job)

   # Start the scheduler
   scheduler.run_pending()

**Decorator Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   @jobs(interval=2, unit='hour', scheduler=scheduler)
   def task_every_2_hours():
       print("Task executed every 2 hours!")

   # Start the scheduler
   scheduler.run_pending()

In this example:

- The job is scheduled to run every 2 hours.
- It will continue running indefinitely until the program is stopped or the job is removed.

Running a Job Every 3 Days
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define the task function
   def task_every_3_days():
       print("Task executed every 3 days!")

   # Create a job to run every 3 days
   job = Job().every(3).day.do(task_every_3_days)

   # Add the job to the scheduler
   scheduler.add_job(job)

   # Start the scheduler
   scheduler.run_pending()

**Decorator Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   @jobs(interval=3, unit='day', scheduler=scheduler)
   def task_every_3_days():
       print("Task executed every 3 days!")

   # Start the scheduler
   scheduler.run_pending()

In this example:

- The job is scheduled to run every 3 days.
- It will continue running every 3 days until the program is stopped or the job is removed.

Running a Job Every 2 Weeks on Monday
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define the task function
   def task_every_2_weeks():
       print("Task executed every 2 weeks!")

   # Create a job to run every 2 weeks on Monday
   job = Job().every(2).week(weekday=0).do(task_every_2_weeks)

   # Add the job to the scheduler
   scheduler.add_job(job)

   # Start the scheduler
   scheduler.run_pending()

**Decorator Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   @jobs(interval=2, unit='week', scheduler=scheduler)
   def task_every_2_weeks():
       print("Task executed every 2 weeks!")

   # Start the scheduler
   scheduler.run_pending()

In this example:

- The job is scheduled to run every 2 weeks on Monday.
- It will continue running every 2 weeks on the specified weekday until the program is stopped or the job is removed.

Running a Job Every 3 Months on the 15th at 9:00 AM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define the task function
   def task_every_3_months():
       print("Task executed every 3 months!")

   # Create a job to run every 3 months on the 15th at 9:00 AM
   job = Job().every(3).month(day=15, time_at="09:00").do(task_every_3_months)

   # Add the job to the scheduler
   scheduler.add_job(job)

   # Start the scheduler
   scheduler.run_pending()

**Decorator Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   @jobs(interval=3, unit='month', time_at="09:00", scheduler=scheduler)
   def task_every_3_months():
       print("Task executed every 3 months!")

   # Start the scheduler
   scheduler.run_pending()

In this example:

- The job is scheduled to run every 3 months on the 15th at 9:00 AM.
- It will continue running every 3 months on the specified date and time until the program is stopped or the job is removed.

Running a Job Every 2 Years on January 1st at Midnight
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Standard Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define the task function
   def task_every_2_years():
       print("Task executed every 2 years!")

   # Create a job to run every 2 years on January 1st at midnight
   job = Job().every(2).year(month=1, day=1, time_at="00:00").do(task_every_2_years)

   # Add the job to the scheduler
   scheduler.add_job(job)

   # Start the scheduler
   scheduler.run_pending()

**Decorator Method:**

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   @jobs(interval=2, unit='year', time_at="00:00", scheduler=scheduler)
   def task_every_2_years():
       print("Task executed every 2 years!")

   # Start the scheduler
   scheduler.run_pending()

In this example:

- The job is scheduled to run every 2 years on January 1st at midnight.
- It will continue running every 2 years on the specified date and time until the program is stopped or the job is removed.

Running Multiple Jobs Simultaneously
####################################

The Job Scheduler allows you to schedule multiple jobs that can run at the same time. For example, you might want to run two different tasks every day at the same time.

Running Multiple Jobs Simultaneously (Standard Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Timeline_Manager import JobScheduler, Job

   # Initialize the scheduler
   scheduler = JobScheduler()

   # Define task functions
   def task1():
       print("Task 1 executed!")

   def task2():
       print("Task 2 executed!")

   # Create jobs that run at the same time
   job1 = Job().every(1).day.at("08:00").do(task1)
   job2 = Job().every(1).day.at("08:00").do(task2)

   # Add jobs to the scheduler
   scheduler.add_job(job1)
   scheduler.add_job(job2)

   # Start the scheduler
   scheduler.run_pending()

Running Multiple Jobs Simultaneously (Decorator Method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from Timeline_Manager import JobScheduler, jobs

   # Initialize the scheduler
   scheduler = JobScheduler()

   @jobs(interval=1, unit='day', time_at="08:00", scheduler=scheduler)
   def task1():
       print("Task 1 executed!")

   @jobs(interval=1, unit='day', time_at="08:00", scheduler=scheduler)
   def task2():
       print("Task 2 executed!")

   # Start the scheduler
   scheduler.run_pending()

In this example:

- Both `task1` and `task2` are scheduled to run at 8:00 AM every day.
- They will execute simultaneously, allowing multiple tasks to be performed at the same time.

Weekday, Day, and Month Ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When scheduling jobs, you can specify specific weekdays, days of the month, and months. Below are the values for each:

- **Weekday Values**:

  - 0: Monday
  - 1: Tuesday
  - 2: Wednesday
  - 3: Thursday
  - 4: Friday
  - 5: Saturday
  - 6: Sunday

- **Day Values**:

  - 1-31: Represents the days of the month.

- **Month Values**:

  - 1: January
  - 2: February
  - 3: March
  - 4: April
  - 5: May
  - 6: June
  - 7: July
  - 8: August
  - 9: September
  - 10: October
  - 11: November
  - 12: December

Threading (Optional)
####################

To run jobs in separate threads (useful for non-blocking execution), set `use_threading=True` when initializing the `JobScheduler`.

.. code-block:: python

   scheduler = JobScheduler(use_threading=True)

In this example:

- The scheduler is set up to run jobs in separate threads, allowing them to execute independently of each other.
- This can be useful for long-running tasks that should not block the execution of other jobs.


