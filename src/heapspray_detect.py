#v3은 v2와 다르게 시스템콜 호출정보는 사용하지 않는다. 오직 kmalloc정보만 사용한다.
#v4는 v3과 다르게 kmalloc과 kfree의 관계를 주소를 확인하지 않고 단순히 개수로 센다.
# -> kmalloc이 5번나오고 kfree가 세번나오면 kmalloc이 두번남아있다고 판단함

#4_2는 로그를 pidtgid로는 구분하지않고 오로지 주소값으로만 구분함
#4_2는 시스템콜 호출수와 실행된 cpu 위치도 구분함

#4_3은 다양한 특징 추출을 함
#4_4는 variance -> entropy로 바뀜
#4_4에서 round,3에 의해서 값이 0이 나타는 현상을 발견 0이 나타나면 log2함수에서 오류가 발생하기 때문에 0.001로 수정하는 update 수행

#4_5부터는 round를 사용하지 않기로 함 raw 데이터를 그대로 사용
#xgboost 훈련 데이터 만드는 코드
#4_6부터는 새로운 feature 시스템콜과 kmalloc의 비를 feature로 넣었음
#4_6부터는 kmalloc_total_count가 1보다 크지 않으면 제거함

#4_7부터는 thread_kmalloc_info에서 kfree, kmalloc_total_count 등을 없애버림
#4_7부터는 kfree에서 에러코드가 있어서 수정하였음 ptr이 0을 넣었던 현상을 발견하였음

#4_8부터는 kfree횟수, kmalloc횟수도 따로 구함
#4_8부터는 kmalloc_count라는 것을 없앴음 무조건 1씩 더함
#4_8부터는 lookup_or_try_init을 syscall호출시작전에만 함
#4_8부터는 syscall 호출 이후 kmalloc_total_count가 0이면 저장하지 않음

#5부터는 kfree횟수, kmalloc횟수도 따로 구함
#5부터는 로그를 사용자 영역에서 집계하는 것이 아니라 커널 영역에서 집계함
#5부터는 미리 슬럽크기별로 변수를 할당해놓고 거기서 일씩 더함

#5_1부터는 ebpf에서 모아놓은 cpu, alloc정보들을 이용해서 entropy를 구하게 함
#5_1에는 cpu_list에 관한 디버깅 코드가 들어가있음
#5_1에는 cpu_list_rate등이 들어가있음

#5_2부터는 cpu_rate는 고려하지않고 cpu의 entropy와 gini계수만 고려할 예정
#5_2에서 last_index부분에 오류가 있어서 추가적으로 수정이 들어감
#5_2에서부터는 kmalloc_total_count가 5이하이면 데이터를 추출하지 않기로함

#5_4에서부터는 heap spray에서 30%이하부터는 검사를 하지 않음

#6에서부터는 필터검사를 넣어서 30%이하는 검사를 하지 않음

#7에서부터는 kmalloc, kfree에 대해서 영수증 시스템을 사용하지 않음 -> 영수증 시스템 사용시 프로그램이 강제종료되는 현상 발생

from time import sleep, strftime
from datetime import datetime
import argparse
import errno
import itertools
import sys
import signal
from bcc import BPF
from bcc.utils import printb
from bcc.syscall import syscall_name, syscalls
import csv
import time
from ctypes import *
import numpy as np
import joblib
from sklearn.feature_extraction import DictVectorizer
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")
adaboost_model = joblib.load('./../data/adaboost_model.joblib')

# signal handler
def signal_ignore(signal, frame):
    print()

def handle_errno(errstr):
    try:
        return abs(int(errstr))
    except ValueError:
        pass

    try:
        return getattr(errno, errstr)
    except AttributeError:
        raise argparse.ArgumentTypeError("couldn't map %s to an errno" % errstr)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--print", action="store_true")
parser.add_argument("--count", action="store_true")
parser.add_argument("-t", "--time", type=float, default=0.05)
parser.add_argument("-l", "--label", type=int, default=0)
parser.add_argument("-n", "--name", type=str, default='poc')
args = parser.parse_args()

text = r"""
#include <linux/sched.h>
#include <uapi/linux/ptrace.h>
#include <linux/cred.h>

#define LAST_SYSCALL_NUMBER 450
#define SCHED_ON 1
#define SCHED_OFF 0
#define PRINT_MODE
#define KMALLOC_ADDRESS_SIZE 30
#define SYSCALL_ON 1
#define SYSCALL_OFF 0

//related to SLUB 
#define HEAP_SRPAY_BOUNDARY 0.3
#define SLUB_COUNT_8 512
#define SLUB_COUNT_16 256
#define SLUB_COUNT_32 128
#define SLUB_COUNT_64 64
#define SLUB_COUNT_96 42
#define SLUB_COUNT_128 32
#define SLUB_COUNT_192 21
#define SLUB_COUNT_256 32
#define SLUB_COUNT_512 32
#define SLUB_COUNT_1K  32
#define SLUB_COUNT_2K  16
#define SLUB_COUNT_4K  8
#define SLUB_COUNT_8K  4 




struct syscall_info{
    u32 pid;
    u32 tid;
    u64 syscall_number;
    u64 count;
    u32 is_not_first;
    char task_name[TASK_COMM_LEN];
};

struct key_prev_syscall_argument{
    u64 pid_tgid;
    u64 syscall_number;
};

struct prev_syscall_argument //이전 직전의 시스템콜변수를 저장하는 변수
{
    u64 prev_args[6];
};

struct syscall_info_key{
    u64 pid;
    u64 syscall_number;
};

struct key_thread_syscall_info
{
    u32 pid;
    u32 tid;
    u64 syscall_number;
};

struct thread_syscall_info
{
    //시스템콜 관련
    u64 syscall_count; //시스템콜 몇번불렀는지
    u64 syscall_argument_similar; //유사한 인자를 넣어서 시스템콜을 호출하였는지 확인
    u64 prev_syscall_number;    //직전 호출한 시스템콜 기록
    u64 syscall_kind_similar;   //시스템콜 유사도

    //출력용
    char task_name[TASK_COMM_LEN];
    u32 pid;
    u32 tid;
    u32 no_name;

    //개별적인 시스템콜
    u64 prev_args[6];
    u64 syscall_number;

    u32 is_first;
};

struct prev_thread_syscall_info
{
    u32 pid;
    u32 tid;
    u64 prev_syscall_number;
    u32 is_first;
};

//파이썬 프로그램에서 가져갈 데이터 총 개수를 위해 사용될 것
struct thread_kmalloc_info_manage{

    char task_name[TASK_COMM_LEN];
    u32 is_first;
    u32 syscall_on;             //시스템콜이 현재 호출되어 있는 상태인지 0 : off 1 : on

    u32 kmalloc_count;
    u32 kfree_count;
    u32 kmalloc_total_count;
    u32 syscall_total_count;

    u32 kmalloc_8;
    u32 kmalloc_16;
    u32 kmalloc_32;
    u32 kmalloc_64;
    u32 kmalloc_96;
    u32 kmalloc_128;
    u32 kmalloc_192;
    u32 kmalloc_256;
    u32 kmalloc_512;
    u32 kmalloc_1K;
    u32 kmalloc_2K;
    u32 kmalloc_4K;
    u32 kmalloc_8K;

    u32 cpu_pos_number[10];
    u32 cpu_pos_number_count[10];
    u32 cpu_pos_number_last_index;

    u32 danger;

    u32 tid;
    u32 tgid;
};

//파이썬 프로그램에서 가져갈 데이터 유사도를 위해 사용될 것
struct thread_kmalloc_info{
    u64 pid_tgid;
    u64 alloc_size;
};

struct key_thread_kmalloc_info{
    u64 pid_tgid;
    u64 ptr_address;            
};

struct kmalloc_address_alert
{
    u32 alert;
    u32 kfree_alert;
};

//시스템콜종류 관련
BPF_HASH(data_syscall_info, struct syscall_info_key, struct syscall_info);
BPF_HASH(data_thread_syscall_info, struct key_thread_syscall_info ,struct thread_syscall_info);
BPF_HASH(data_prev_thread_syscall_info, u64, struct prev_thread_syscall_info);
BPF_HASH(data_thread_kmalloc_info_manage, u64, struct thread_kmalloc_info_manage);
//BPF_HASH(data_thread_kmalloc_info, struct key_thread_kmalloc_info, struct thread_kmalloc_info);
BPF_HASH(data_thread_kmalloc_info, u64 , struct thread_kmalloc_info);       //address단위로 구분
BPF_HASH(data_kmalloc_address_alert, u64, struct kmalloc_address_alert);

/*
TRACEPOINT_PROBE(raw_syscalls, sys_enter) {
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    const struct cred * cred = task->cred;
    if((cred->euid).val == 0)
    {
        goto done;
    }
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = (u32)pid_tgid;
    u32 syscall_argument_similar = 0;

    struct thread_kmalloc_info_manage * val_thread_kmalloc_info_manage, zero_val_thread_kmalloc_info_manage = {};
    val_thread_kmalloc_info_manage = data_thread_kmalloc_info_manage.lookup_or_try_init(&pid_tgid, &zero_val_thread_kmalloc_info_manage);
    //kmalloc 관련
    if(val_thread_kmalloc_info_manage)
    {
        if(val_thread_kmalloc_info_manage->syscall_on == SYSCALL_OFF)
        {
            val_thread_kmalloc_info_manage->syscall_on = SYSCALL_ON;    //1일 때만 kmalloc에 대해서 검사할 것임
            if(val_thread_kmalloc_info_manage->is_first == 0)
            {
                char name[TASK_COMM_LEN];
                bpf_get_current_comm(&name, sizeof(name));
                bpf_probe_read_str((char*)val_thread_kmalloc_info_manage->task_name,sizeof(name),name);
                val_thread_kmalloc_info_manage->is_first = 1;
                val_thread_kmalloc_info_manage->tgid = pid;
                val_thread_kmalloc_info_manage->tid = tid;
            }
            val_thread_kmalloc_info_manage->syscall_total_count += 1;
        }
    }
done :
    return 0;
}
*/

TRACEPOINT_PROBE(kmem, kmalloc){
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    const struct cred * cred = task->cred;
    /*
    if((cred->euid).val == 0)
    {
        goto done;
    }
    */
    if(task->flags & PF_KTHREAD)
    {   
        bpf_trace_printk("kernel thread");
        goto done;
    }

    struct thread_kmalloc_info_manage * val_thread_kmalloc_info_manage, zero_val_thread_kmalloc_info_manage = {};
    struct thread_kmalloc_info * val_thread_kmalloc_info, zero_val_thread_kmalloc_info = {};
    u64 ptr_address = (u64)args->ptr;
    u64 alloc_size = args->bytes_alloc;
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u32 pid = pid_tgid >> 32;
    u32 tid = (u32)pid_tgid;

    val_thread_kmalloc_info = data_thread_kmalloc_info.lookup_or_try_init(&ptr_address, &zero_val_thread_kmalloc_info);
    if(val_thread_kmalloc_info)
    {
        if(val_thread_kmalloc_info->alloc_size == 0)
        {
            val_thread_kmalloc_info->pid_tgid = pid_tgid;
            val_thread_kmalloc_info->alloc_size = alloc_size;
        }
        else
        {
            bpf_trace_printk("kmalloc error[0] %ul",ptr_address);
            goto done;
        }
    }
    else
    {
        bpf_trace_printk("kmalloc error[0]");
        goto done;
    }

    val_thread_kmalloc_info_manage = data_thread_kmalloc_info_manage.lookup_or_try_init(&pid_tgid,&zero_val_thread_kmalloc_info_manage);
    if(val_thread_kmalloc_info_manage)
    {
        if(val_thread_kmalloc_info_manage->is_first == 0)
        {
            char name[TASK_COMM_LEN];
            bpf_get_current_comm(&name, sizeof(name));
            bpf_probe_read_str((char*)val_thread_kmalloc_info_manage->task_name,sizeof(name),name);
            val_thread_kmalloc_info_manage->is_first = 1;
            val_thread_kmalloc_info_manage->tgid = pid;
            val_thread_kmalloc_info_manage->tid = tid;
        }
        val_thread_kmalloc_info_manage->kmalloc_count += 1;
        val_thread_kmalloc_info_manage->kmalloc_total_count += 1;
        if(alloc_size == 8)
        {
            val_thread_kmalloc_info_manage->kmalloc_8 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_8 > SLUB_COUNT_8 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<0;
            }
        }
        else if(alloc_size == 16)
        {
            val_thread_kmalloc_info_manage->kmalloc_16 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_16 > SLUB_COUNT_16 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<1;
            }
        }
        else if(alloc_size == 32)
        {
            val_thread_kmalloc_info_manage->kmalloc_32 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_32 > SLUB_COUNT_32 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<2;
            }
        }
        else if(alloc_size == 64)
        {
            val_thread_kmalloc_info_manage->kmalloc_64 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_64 > SLUB_COUNT_64 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<3;
            }
        }
        else if(alloc_size == 96)
        {
            val_thread_kmalloc_info_manage->kmalloc_96 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_96 > SLUB_COUNT_96 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<4;
            }
        }
        else if(alloc_size == 128)
        {
            val_thread_kmalloc_info_manage->kmalloc_128 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_128 > SLUB_COUNT_128 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<5;
            }
        }
        else if(alloc_size == 192)
        {
            val_thread_kmalloc_info_manage->kmalloc_192 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_192 > SLUB_COUNT_192 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<6;
            }
        }
        else if(alloc_size == 256)
        {
            val_thread_kmalloc_info_manage->kmalloc_256 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_256 > SLUB_COUNT_256 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<7;
            }
        }
        else if(alloc_size == 512)
        {
            val_thread_kmalloc_info_manage->kmalloc_512 += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_512 > SLUB_COUNT_512 * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<8;
            }
        }
        else if(alloc_size == 1024)
        {
            val_thread_kmalloc_info_manage->kmalloc_1K += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_1K > SLUB_COUNT_1K * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<9;
            }
        }
        else if(alloc_size == 2048)
        {
            val_thread_kmalloc_info_manage->kmalloc_2K += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_2K > SLUB_COUNT_2K * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<10;
            }
        }
        else if(alloc_size == 4096)
        {
            val_thread_kmalloc_info_manage->kmalloc_4K += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_4K > SLUB_COUNT_4K * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<11;
            }
        }
        else if(alloc_size == 8192)
        {
            val_thread_kmalloc_info_manage->kmalloc_8K += 1;
            if(val_thread_kmalloc_info_manage->kmalloc_8K > SLUB_COUNT_8K * HEAP_SRPAY_BOUNDARY)
            {
                val_thread_kmalloc_info_manage->danger |= 1<<12;
            }
        }
    }
done :
    return 0;
}

TRACEPOINT_PROBE(kmem, kfree){
    u64 pid_tgid = bpf_get_current_pid_tgid();
    u64 alloc_size = 0;

    struct thread_kmalloc_info_manage * val_thread_kmalloc_info_manage, zero_val_thread_kmalloc_info_manage = {};
    struct thread_kmalloc_info * val_thread_kmalloc_info, zero_val_thread_kmalloc_info = {};

    u64 ptr_address = (u64)args->ptr;
    val_thread_kmalloc_info = data_thread_kmalloc_info.lookup(&ptr_address);
    if(val_thread_kmalloc_info)
    {
        pid_tgid = val_thread_kmalloc_info->pid_tgid;
        alloc_size = val_thread_kmalloc_info->alloc_size;
        data_thread_kmalloc_info.delete(&ptr_address);
    }
    else
    {
        struct task_struct *task = (struct task_struct *)bpf_get_current_task();
        const struct cred * cred = task->cred;
        if((cred->euid).val == 0)
        {
            goto done;
        }
        if(task->flags & PF_KTHREAD)
        {   
            bpf_trace_printk("kernel thread free[0]");
            goto done;
        }
        if(ptr_address == 0)
        {
            goto done;
        }
        bpf_trace_printk("kfree error[0] %ul",ptr_address);
        goto done;
    }
    val_thread_kmalloc_info_manage = data_thread_kmalloc_info_manage.lookup(&pid_tgid);
    if(val_thread_kmalloc_info_manage)
    {
        val_thread_kmalloc_info_manage->kfree_count += 1;
        if(val_thread_kmalloc_info_manage->kmalloc_total_count> 0)
        {
            val_thread_kmalloc_info_manage->kmalloc_total_count -= 1;
        }
        if(alloc_size == 8)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_8 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_8 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_8 <= SLUB_COUNT_8 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<0);
                }
            }
        }
        else if(alloc_size == 16)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_16 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_16 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_16 <= SLUB_COUNT_16 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<1);
                }
            }
        }
        else if(alloc_size == 32)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_32 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_32 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_32 <= SLUB_COUNT_32 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<2);
                }
            }
        }
        else if(alloc_size == 64)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_64 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_64 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_64 <= SLUB_COUNT_64 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<3);
                }
            }
        }
        else if(alloc_size == 96)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_96 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_96 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_96 <= SLUB_COUNT_96 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<4);
                }
            }
        }
        else if(alloc_size == 128)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_128 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_128 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_128 <= SLUB_COUNT_128 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<5);
                }
            }
        }
        else if(alloc_size == 192)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_192 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_192 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_192 <= SLUB_COUNT_192 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<6);
                }
            }
        }
        else if(alloc_size == 256)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_256 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_256 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_256 <= SLUB_COUNT_256 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<7);
                }
            }
        }
        else if(alloc_size == 512)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_512 > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_512 -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_512 <= SLUB_COUNT_512 * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<8);
                }
            }
        }
        else if(alloc_size == 1024)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_1K > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_1K -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_1K <= SLUB_COUNT_1K * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<9);
                }
            }
        }
        else if(alloc_size == 2048)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_2K > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_2K -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_2K <= SLUB_COUNT_2K * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<10);
                }
            }
        }
        else if(alloc_size == 4096)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_4K > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_4K -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_4K <= SLUB_COUNT_4K * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<11);
                }
            }
        }
        else if(alloc_size == 8192)
        {
            if(val_thread_kmalloc_info_manage->kmalloc_8K > 0)
            {
                val_thread_kmalloc_info_manage->kmalloc_8K -= 1;
                if(val_thread_kmalloc_info_manage->kmalloc_8K <= SLUB_COUNT_8K * HEAP_SRPAY_BOUNDARY)
                {
                    val_thread_kmalloc_info_manage->danger &= ~(1<<12);
                }
            }
        }
    }
    else
    {
        struct task_struct *task = (struct task_struct *)bpf_get_current_task();
        const struct cred * cred = task->cred;
        if((cred->euid).val == 0)
        {
            goto done;
        }
        bpf_trace_printk("kfree error[1] %u %u",pid_tgid >> 32, (u32)pid_tgid);
    }
done :
    return 0;
}

int kprobe__do_exit(void *ctx) {
    struct task_struct *task = (struct task_struct *)bpf_get_current_task();
    const struct cred * cred = task->cred;
    /*
    if((cred->euid).val == 0)
    {
        goto done;
    }
    */
    u64 pid_tgid = bpf_get_current_pid_tgid();
    
    data_thread_kmalloc_info_manage.delete(&pid_tgid);
done :    
    return 0;
}

"""

if args.count:
    text = ("#define COUNT\n" + text)

file_name_ori = input('파일 이름을 입력해주세요: ')

bpf = BPF(text=text)
continue_count = 0

def print_process_info():
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),'start')
    print_data_thread_kmalloc_info = bpf["data_thread_kmalloc_info"]
    print_data_thread_kmalloc_info_manage = bpf["data_thread_kmalloc_info_manage"]
    global print_type
    global print_line_count
    global continue_count
    global before_filter_count
    global after_filter_count
    #collect_print_data_thread_kmalloc_info = print_data_thread_kmalloc_info.items_lookup_and_delete_batch()
    collect_print_data_thread_kmalloc_info_manage = print_data_thread_kmalloc_info_manage.items_lookup_and_delete_batch()
    print_data_thread_kmalloc_info.items_delete_batch()
    #print_data_thread_kmalloc_info.clear()
    dictionary_collect_print_data_syscall_info = {}
    dictionary_collect_print_data_syscall_info_list = []
    pid_tgid_list = []   

    #filter_collect_print_data_thread_kmalloc_info_manage = make_generator(collect_print_data_thread_kmalloc_info_manage)
    #print(before_filter_count,after_filter_count)
    before_filter_count = 0
    after_filter_count = 0
    for k, v in collect_print_data_thread_kmalloc_info_manage:
        #before_filter_count += 1

        if v.danger == 0:
            continue
        
        #if v.kmalloc_total_count == 0:
        #    continue   
        """
        else:
            print((v.task_name).decode('utf-8'),'danger')
        """
        #after_filter_count += 1
        pid_tgid_list.append(k.value)
        if dictionary_collect_print_data_syscall_info.get(k.value) == None:
            dictionary_collect_print_data_syscall_info[k.value] = {}
        else:
            print('why two object?', k.value)
            continue
        dict_cur = dictionary_collect_print_data_syscall_info[k.value]
        dict_cur['tid'] = v.tid
        dict_cur['tgid'] = v.tgid
        dict_cur['process_name'] = (v.task_name).decode('utf-8')
        dict_cur['kmalloc_invocate_list'] = []
        dict_cur['kmalloc_total_count'] = v.kmalloc_total_count
        dict_cur['kmalloc_kind'] = 0    
        dict_cur['kmalloc_top'] = []
        dict_cur['address_list'] = []
        dict_cur['kmalloc_top_rate'] = []
        dict_cur['kmalloc_top_entropy'] = 0
        dict_cur['kmalloc_top_gini'] = 0
        dict_cur['kfree_count'] = v.kfree_count
        dict_cur['kmalloc_count'] = v.kmalloc_count
        dict_cur['danger'] = v.danger

        if v.kmalloc_8 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_16 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_32 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_64 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_96 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_128 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_192 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_256 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_512 != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_1K != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_2K != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_4K != 0:
            dict_cur['kmalloc_kind'] += 1
        if v.kmalloc_8K != 0:
            dict_cur['kmalloc_kind'] += 1
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_8,8])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_16,16])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_32,32])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_64,64])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_96,96])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_128,128])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_192,192])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_256,256])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_512,512])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_1K,1024])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_2K,2048])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_4K,4096])
        dict_cur['kmalloc_invocate_list'].append([v.kmalloc_8K,8192])
        sorted_list = sorted(dict_cur['kmalloc_invocate_list'], key=lambda x: -x[0])
        dict_cur['kmalloc_invocate_list'] = sorted_list
        for i in range(0, dict_cur['kmalloc_kind']):
            dict_cur['kmalloc_top'].append(dict_cur['kmalloc_invocate_list'][i][0])
        for i in range(0, dict_cur['kmalloc_kind']):
            dict_cur['kmalloc_top_rate'].append(dict_cur['kmalloc_top'][i]/sum(dict_cur['kmalloc_top']))
        if len(dict_cur['kmalloc_top_rate']) != 0:
            kmalloc_top_rate_entropy = np.array(dict_cur['kmalloc_top_rate'])
            dict_cur['kmalloc_top_entropy'] =  (-kmalloc_top_rate_entropy*np.log2(kmalloc_top_rate_entropy)).sum()
            dict_cur['kmalloc_top_gini'] = 1 - ((kmalloc_top_rate_entropy / kmalloc_top_rate_entropy.sum()) ** 2).sum()
        """
        if dict_cur['kmalloc_kind'] < 3:
            for i in range(0, dict_cur['kmalloc_kind']):
                dict_cur['kmalloc_top'].append(dict_cur['kmalloc_invocate_list'][i][0])
        else :
            for i in range(0, 3):
                dict_cur['kmalloc_top'].append(dict_cur['kmalloc_invocate_list'][i][0])
        if len(dict_cur['kmalloc_top']) < 3:
            for i in range( 0 , len(dict_cur['kmalloc_top'])):
                dict_cur['kmalloc_top_rate'].append(dict_cur['kmalloc_top'][i]/sum(dict_cur['kmalloc_top']))
        else:
            for i in range( 0 , 3):
                dict_cur['kmalloc_top_rate'].append(dict_cur['kmalloc_top'][i]/sum(dict_cur['kmalloc_top']))
        if len(dict_cur['kmalloc_top_rate']) != 0:
            kmalloc_top_rate_entropy = np.array(dict_cur['kmalloc_top_rate'])
            dict_cur['kmalloc_top_entropy'] =  (-kmalloc_top_rate_entropy*np.log2(kmalloc_top_rate_entropy)).sum()
            dict_cur['kmalloc_top_gini'] = 1 - ((kmalloc_top_rate_entropy / kmalloc_top_rate_entropy.sum()) ** 2).sum()
        for i in range(len(dict_cur['kmalloc_top']), 3):
            dict_cur['kmalloc_invocate_list'].append([0,0])
            dict_cur['kmalloc_top'].append((0))
            dict_cur['kmalloc_top_rate'].append(0)
        dict_cur['kmalloc_top1'] = dict_cur['kmalloc_top'][0]
        dict_cur['kmalloc_top2'] = dict_cur['kmalloc_top'][1]
        dict_cur['kmalloc_top3'] = dict_cur['kmalloc_top'][2]
        dict_cur['kamlloc_top1_rate'] = dict_cur['kmalloc_top_rate'][0]
        dict_cur['kamlloc_top2_rate'] = dict_cur['kmalloc_top_rate'][1]
        dict_cur['kamlloc_top3_rate'] = dict_cur['kmalloc_top_rate'][2]
        """
    """
    danger = []
    if len(dictionary_collect_print_data_syscall_info.items()) != 0:
        vectorizer = DictVectorizer(sparse=False)
        X = vectorizer.fit_transform(dictionary_collect_print_data_syscall_info.items())
        danger = adaboost_model.predict(X)
    

    for el in danger:
        print(el)
    #데이터 생성 완성
    """

    print_check = 0
    if args.label == 0:
        for k, v in dictionary_collect_print_data_syscall_info.items():
            log_data = np.array([[v['kmalloc_total_count'], v['kmalloc_count'], v['kfree_count'], v['kmalloc_kind'], v['kmalloc_top_entropy'], v['kmalloc_top_gini']]])
            danger = adaboost_model.predict(log_data)
            print(v['process_name'],danger)

    #print(before_filter_count,after_filter_count)
    """
    if print_check == 0:
        write_data = [print_type]
        writer.writerow(write_data)
        print_line_count += 1
    """
    print_type += 1
    #print(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f'),'end')
print_line_count = 0        #현재 몇줄까지 입력되어 있는지 저장하는 변수
print_line_check = 100000   #몇줄까지는 입력할 수 있는지 정하는 변수
print_type = 0                   
file_number = 0             #현재 몇번째 파일로 저장하는지 지정하는 변수
file_name = file_name_ori +'_'+str(file_number)

before_filter_count = 0
after_filter_count = 0

f = open(file_name+'.csv', 'w')
writer = csv.writer(f)

print_type = 0
is_print = 0
first_time = 0
exiting = 0
print('start')
title_data = []
if args.label == -1:
    title_data = ['PRINT_TYPE', 'TGID', 'TID', 'PROCESS_NAME', 'KMALLOC_TOTAL_COUNT', 'KMALLOC_COUNT','KFREE_COUNT','KMALLOC_KIND', 'KMALLOC_TOP_ENTROPY','KMALLOC_TOP_GINI','DANGER']
elif args.label == 0 or args.label == 1:
    title_data = ['PRINT_TYPE', 'TGID', 'TID', 'PROCESS_NAME', 'KMALLOC_TOTAL_COUNT', 'KMALLOC_COUNT','KFREE_COUNT','KMALLOC_KIND', 'KMALLOC_TOP_ENTROPY','KMALLOC_TOP_GINI','DANGER']
writer.writerow(title_data)
print_line_count += 1
while True:
    try:
        if args.count:
            if print_line_count > print_line_check:         #특정 줄수 이상을 넘으면 파일을 닫고 새로운 파일을 만든다.
                print_line_count = 0
                f.close()
                file_number += 1
                file_name = file_name_ori +'_'+ str(file_number)
                f = open(file_name+'.csv', 'w')
                writer = csv.writer(f)
                writer.writerow(title_data)
                print_line_count += 1
        else:
            print_process_info()
            if print_line_count > print_line_check:         #특정 줄수 이상을 넘으면 파일을 닫고 새로운 파일을 만든다.
                print_line_count = 0
                f.close()
                file_number += 1
                file_name = file_name_ori +'_'+ str(file_number)
                f = open(file_name+'.csv', 'w')
                writer = csv.writer(f)
                writer.writerow(title_data)
                print_line_count += 1
        sleep(args.time)
    except KeyboardInterrupt:
        exiting = 1
        signal.signal(signal.SIGINT, signal_ignore)
    if exiting:
        #f.close()
        print("Detaching...")
        exit()
