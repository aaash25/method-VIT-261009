"""NOT DEPLOYABLE: CPU prototype for quality-only, closed-epoch immutable archives.

The caller runs the unchanged entry with a600-second inner budget and a bounded
outer supervisor, waits for its actual return code, and supplies it to run_epoch.
No model loading, subprocess launching, timing hook, deletion, rename, or retry
is implemented here. Receipt bytes are recovery candidates, never live ACKs.
"""
from contextlib import contextmanager
import base64
import fcntl
import gzip
import hashlib
import io as bytesio
import json
import os
from pathlib import Path
import re
import stat
import uuid
import zlib

MAX_TREE = 128 * 1024 * 1024
MAX_DELTA = 32 * 1024 * 1024
MAX_CONTAINER = 64 * 1024 * 1024

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def close_after_error(resource,primary,closer=None):
    try:
        (closer or resource.close)()
    except BaseException as secondary:
        try:
            primary.add_note('Secondary close failure: '+repr(secondary))
        except BaseException:
            pass

@contextmanager
def closing(resource,closer=None):
    try:
        yield resource
    except BaseException as primary:
        close_after_error(resource,primary,closer)
        raise
    else:
        (closer or resource.close)()

class IO:
    """Real small file operations; optional explicit test fault hook."""
    def __init__(self, hook=None):
        self.hook=hook or (lambda stage,path:None)
        self.events=[]
    def point(self,stage,path=None):
        self.events.append((stage,str(path) if path is not None else None))
        self.hook(stage,path)
    def sync_dir(self,path,label):
        self.point(label+':dir_fsync',path)
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
        with closing(fd,lambda:os.close(fd)):os.fsync(fd)
    def write(self,path,raw,label):
        with closing(path.open('xb')) as stream:
            self.point(label+':write',path)
            if stream.write(raw)!=len(raw):raise OSError('short archive write')
            self.point(label+':flush',path);stream.flush()
            self.point(label+':fsync',path);os.fsync(stream.fileno())
        self.sync_dir(path.parent,label)
        self.point(label+':readback',path)
        if sha(path.read_bytes())!=sha(raw):raise RuntimeError('archive readback hash mismatch')
        return sha(raw)
    def resync(self,path,expected,label):
        if sha(path.read_bytes())!=expected:raise RuntimeError('recovery candidate hash mismatch')
        with closing(path.open('rb')) as stream:
            self.point(label+':fsync',path);os.fsync(stream.fileno())
        self.sync_dir(path.parent,label)
        if sha(path.read_bytes())!=expected:raise RuntimeError('recovery candidate changed during sync')
    def mkdir(self,path):
        path.mkdir(mode=0o700,exist_ok=False)
        self.sync_dir(path.parent,'new_directory')

def checked_binding(expected,ids):
    if expected.get('schema')!='candidate_full_quality_v1' or expected.get('kind')!='candidate_only_quality' or expected.get('timing_generation') is not False or expected.get('dense_generation') is not False:
        raise RuntimeError('quality-only frozen contract required')
    required=('source_seal','source_hashes','frozen_method_sources','entry_sha256','snapshot','data_root','model','revision','protocol','manifest_digests','method','method_point','dtype','gpu','job','phase')
    if any(name not in expected for name in required):raise RuntimeError('incomplete scientific contract binding')
    if set(ids)!=set(expected['manifest_digests']) or any(len(v)!=len(set(v)) or any(not isinstance(x,str) or not x for x in v) for v in ids.values()) or sum(map(len,ids.values()))!=expected['expected_units']:
        raise RuntimeError('manifest identifier binding mismatch')
    return sha(canonical({'contract':expected,'expected_ids':ids}))

def collect(job):
    result={};total=0
    if not job.exists():return result
    for path in sorted(job.rglob('*')):
        relative=path.relative_to(job)
        if any(part in ('scratch','tmp') for part in relative.parts) or path.name=='writer.lock':continue
        value=path.lstat()
        if stat.S_ISDIR(value.st_mode):continue
        if not stat.S_ISREG(value.st_mode):raise RuntimeError('non-regular output evidence')
        if relative.parts[0] not in ('CONTRACT.json','RUNTIME.json','journals','events','quality_exports'):
            raise RuntimeError('unexpected output evidence')
        total+=value.st_size
        if total>MAX_TREE or len(result)>=4096:raise RuntimeError('bounded prototype output limit exceeded')
        raw=path.read_bytes()
        if len(raw)!=value.st_size:raise RuntimeError('output changed during read')
        result[str(relative)]=raw
    return result

def hashes(files):return {name:sha(raw) for name,raw in sorted(files.items())}

def validate_success(previous,files,expected,ids,rc):
    if rc not in (0,75):raise RuntimeError('epoch did not exit normally0/75')
    current=hashes(files)
    if any(current.get(name)!=digest for name,digest in previous.items()):raise RuntimeError('previous immutable output changed')
    try:
        if json.loads(files['CONTRACT.json'])!=expected:raise RuntimeError('contract changed')
        json.loads(files['RUNTIME.json'])
    except (KeyError,ValueError) as error:raise RuntimeError('missing or incomplete contract/runtime evidence') from error
    new_names=set(files)-set(previous)
    events=[json.loads(files[name]) for name in sorted(new_names) if name.startswith('events/')]
    starts=[event for event in events if event.get('event')=='dense_reference']
    finals=[event for event in events if event.get('event')=='final']
    if len(starts)!=1 or len(finals)!=1:raise RuntimeError('exactly one new epoch start/final required')
    final=finals[0];epoch=starts[0].get('epoch')
    if not epoch or final.get('epoch')!=epoch or any(event.get('epoch')!=epoch or event.get('event')=='failure' for event in events):
        raise RuntimeError('epoch evidence mismatch or failure event')
    desired='generation_complete' if rc==0 else 'paused_budget'
    if final.get('state')!=desired:raise RuntimeError('exit code/final state mismatch')
    for field in ('schema','phase','job','method','dtype','expected_units'):
        if final.get(field)!=expected[field]:raise RuntimeError('final binding mismatch: '+field)
    if final.get('dense_generated_units')!=0 or final.get('timing_generated_units')!=0:
        raise RuntimeError('not independent candidate quality')
    for name in files:
        if name.startswith('journals/') and not any(name=='journals/quality/candidate/'+dataset+'/contract.json' or re.fullmatch(re.escape('journals/quality/candidate/'+dataset+'/')+r'segment_\d{6}\.jsonl',name) for dataset in ids):
            raise RuntimeError('unbound journal evidence')
    for dataset,receipt in final.get('exports',{}).items():
        if dataset not in ids:raise RuntimeError('unexpected export dataset')
        name='quality_exports/'+dataset+'/'+Path(receipt['path']).name
        receipt_name=str(Path(name).with_suffix('.receipt.json'))
        if name not in files or receipt_name not in files or sha(files[name])!=receipt['sha256'] or json.loads(files[receipt_name])!=receipt or receipt['rows']!=len(ids[dataset]) or receipt['manifest_digest']!=expected['manifest_digests'][dataset]:
            raise RuntimeError('export coverage/hash/receipt mismatch')
    count=0
    for dataset,allowed in ids.items():
        prefix='journals/quality/candidate/'+dataset+'/'
        if json.loads(files[prefix+'contract.json'])!=expected|{'dataset':dataset,'mode':'candidate'}:
            raise RuntimeError('dataset contract mismatch')
        names=sorted(name for name in files if name.startswith(prefix+'segment_'))
        if names!=[prefix+f'segment_{i:06d}.jsonl' for i in range(len(names))]:raise RuntimeError('segment gap or invalid name')
        seen=set()
        for name in names:
            for line in files[name].splitlines(keepends=True):
                if not line.endswith(b'\n'):raise RuntimeError('uncertain incomplete journal tail')
                envelope=json.loads(line);record=envelope['record'];key=record['key'];value=record['result']
                if envelope['crc32']!=f'{zlib.crc32(canonical(record)):08x}':raise RuntimeError('journal CRC mismatch')
                if key not in allowed or key in seen:raise RuntimeError('duplicate or unexpected record key')
                if (value.get('dataset'),value.get('request_id'),value.get('mode'))!=(dataset,key,'candidate'):
                    raise RuntimeError('record identity mismatch')
                if name in new_names and value.get('epoch')!=epoch:raise RuntimeError('new segment contains another epoch')
                seen.add(key);count+=1
    if count!=final['committed_units'] or (rc==0 and count!=expected['expected_units']) or (rc==75 and count>=expected['expected_units']):
        raise RuntimeError('final coverage/count mismatch')
    return final,current,sorted(new_names),count

def unpack(raw):
    if len(raw)>MAX_CONTAINER:raise RuntimeError('archive too large')
    with gzip.GzipFile(fileobj=bytesio.BytesIO(raw)) as stream:body=stream.read(MAX_CONTAINER+1)
    if len(body)>MAX_CONTAINER:raise RuntimeError('expanded archive too large')
    payload=json.loads(body);files={};total=0
    for name,encoded in payload['files'].items():
        p=Path(name)
        if p.is_absolute() or '..' in p.parts:raise RuntimeError('unsafe archived path')
        value=base64.b64decode(encoded,validate=True);total+=len(value)
        if total>MAX_DELTA or len(files)>=4096:raise RuntimeError('archive delta too large')
        files[name]=value
    return payload,files

class Coordinator:
    @classmethod
    def create(cls,local,shared,expected,ids,io=None):
        self=cls();self.io=io or IO();self.local=Path(local);self.shared=Path(shared)
        self.expected=json.loads(canonical(expected));self.ids=json.loads(canonical(ids));self.binding=checked_binding(self.expected,self.ids)
        self.io.mkdir(self.local);self.io.mkdir(self.shared)
        self.lock=(self.shared/'coordinator.lock').open('xb+')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.session={'schema':'quality_epoch_archive_v1','run_id':uuid.uuid4().hex,'binding':self.binding}
            raw=canonical(self.session)+b'\n'
            self.io.write(self.shared/'SESSION.json',raw,'shared_session')
            self.io.write(self.local/'SESSION.json',raw,'local_session')
        except BaseException as primary:
            close_after_error(self.lock,primary);raise
        self._ready({},0,None,0)
        return self
    def _ready(self,snapshot,seq,head,count):
        self.output=self.local/'output'
        self.job=self.output/self.expected['schema']/self.expected['phase']/f"job_{self.expected['job']}"
        self.snapshot=snapshot;self.seq=seq;self.head=head;self.count=count;self.state='READY' if seq==0 else 'ACKED'
    def close(self):
        try:self.lock.close()
        except BaseException:
            self.state='HALTED';raise
        self.state='CLOSED'
    def run_epoch(self,run):
        if self.state not in ('READY','ACKED') or self.lock.closed:raise RuntimeError('coordinator halted/closed; explicit fresh restore required')
        try:
            if hashes(collect(self.job))!=self.snapshot:raise RuntimeError('uncertain local OUT; do not reopen')
            self.state='RUNNING';rc=run(self.job)
            files=collect(self.job)
            final,snapshot,new,count=validate_success(self.snapshot,files,self.expected,self.ids,rc)
            self.state='ELIGIBLE'
            delta={name:files[name] for name in new}
            if sum(map(len,delta.values()))>MAX_DELTA:raise RuntimeError('delta exceeds prototype bound')
            payload=dict(schema='quality_epoch_archive_v1',run_id=self.session['run_id'],binding=self.binding,
                         seq=self.seq,previous_receipt_sha256=self.head,exit_code=rc,epoch=final['epoch'],
                         count=count,snapshot=snapshot,files={name:base64.b64encode(raw).decode() for name,raw in delta.items()})
            body=canonical(payload)
            if len(body)>MAX_CONTAINER:raise RuntimeError('encoded batch exceeds bound')
            archive=f'batch_{self.seq:06d}_{uuid.uuid4().hex}.json.gz'
            archive_hash=self.io.write(self.shared/archive,gzip.compress(body,mtime=0),'archive')
            self.state='ARCHIVE_DURABLE';self.io.point('before_receipt',self.shared/archive)
            receipt=f'receipt_{self.seq:06d}_{uuid.uuid4().hex}.json'
            info=dict(schema='archive_receipt_candidate_v1',run_id=self.session['run_id'],binding=self.binding,
                      seq=self.seq,previous_receipt_sha256=self.head,archive=archive,archive_sha256=archive_hash,
                      epoch=final['epoch'],count=count,final_state=final['state'],snapshot=snapshot,new_files=new)
            digest=self.io.write(self.shared/receipt,canonical(info)+b'\n','receipt')
            self.io.point('after_receipt',self.shared/receipt)
            # This transition is an observed successful operation, not a disk scan.
            self.snapshot=snapshot;self.head=digest;self.seq+=1;self.count=count;self.state='ACKED'
            return dict(info,receipt=receipt,receipt_sha256=digest)
        except BaseException:
            self.state='HALTED';raise
    @classmethod
    def restore(cls,shared,local,expected,ids,receipts,*,resync=False,io=None):
        if not resync or not receipts:raise RuntimeError('explicit nonempty chain revalidation/re-sync required')
        self=cls();self.io=io or IO();self.shared=Path(shared);self.local=Path(local)
        if self.local.exists():raise FileExistsError(self.local)
        self.expected=json.loads(canonical(expected));self.ids=json.loads(canonical(ids));self.binding=checked_binding(self.expected,self.ids)
        self.lock=(self.shared/'coordinator.lock').open('r+b')
        try:
            fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            session_raw=(self.shared/'SESSION.json').read_bytes();self.session=json.loads(session_raw)
            if self.session['binding']!=self.binding:raise RuntimeError('recovery binding changed')
            if set(receipts)!={path.name for path in self.shared.glob('receipt_*.json')} or len(receipts)!=len(set(receipts)):
                raise RuntimeError('ambiguous receipt set; do not append a branch behind an existing head')
            self.io.resync(self.shared/'SESSION.json',sha(session_raw),'recover_session')
            files={};snapshot={};head=None;count=0
            for seq,name in enumerate(receipts):
                if Path(name).name!=name:raise RuntimeError('receipt must be an explicit shared-root basename')
                raw=(self.shared/name).read_bytes();info=json.loads(raw)
                if info['schema']!='archive_receipt_candidate_v1' or info['run_id']!=self.session['run_id'] or info['binding']!=self.binding or info['seq']!=seq or info['previous_receipt_sha256']!=head:
                    raise RuntimeError('receipt gap, branch, duplicate, or changed binding')
                archive=info['archive']
                if Path(archive).name!=archive:raise RuntimeError('unsafe archive reference')
                packed=(self.shared/archive).read_bytes()
                if sha(packed)!=info['archive_sha256']:raise RuntimeError('archive hash mismatch')
                payload,delta=unpack(packed)
                if payload.get('schema')!='quality_epoch_archive_v1':raise RuntimeError('unknown archive schema')
                for field in ('run_id','binding','seq','previous_receipt_sha256','epoch','count','snapshot'):
                    if payload[field]!=info[field]:raise RuntimeError('archive/receipt mismatch')
                if set(delta)&set(files):raise RuntimeError('archive repeats an already materialized file')
                merged=files|delta
                final,actual,new,count=validate_success(snapshot,merged,self.expected,self.ids,payload['exit_code'])
                if actual!=info['snapshot'] or new!=info['new_files'] or count!=info['count'] or final['epoch']!=info['epoch'] or final['state']!=info['final_state']:
                    raise RuntimeError('restored manifest/coverage mismatch')
                self.io.resync(self.shared/archive,info['archive_sha256'],'recover_archive')
                self.io.resync(self.shared/name,sha(raw),'recover_receipt')
                files=merged;snapshot=actual;head=sha(raw)
            # All selected candidates have NOW passed sync, hash and binding checks.
            self.io.mkdir(self.local)
            self.io.write(self.local/'SESSION.json',session_raw,'restore_session')
            job=self.local/'output'/self.expected['schema']/self.expected['phase']/f"job_{self.expected['job']}"
            for name,raw in files.items():
                target=job/name;missing=[];parent=target.parent
                while not parent.exists():
                    missing.append(parent);parent=parent.parent
                for directory in reversed(missing):self.io.mkdir(directory)
                self.io.write(target,raw,'restore_local')
            self._ready(snapshot,len(receipts),head,count)
            return self
        except BaseException as primary:
            close_after_error(self.lock,primary);raise
