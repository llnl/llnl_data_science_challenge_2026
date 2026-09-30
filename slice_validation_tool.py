"""Memory-safe Brian CT slice validation tool; run with --slice 446."""
from pathlib import Path
import argparse, csv, json
import numpy as np
import tifffile
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parent
CT=ROOT/'data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif'
STAGE=ROOT/'part2/stage_2a_developer_output'
CAD=ROOT/'data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json'
OUT=ROOT/'outputs/brian_defect_overlay/slice_validation'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--slice',type=int,default=446,dest='z'); a=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    ct=tifffile.memmap(CT); z=a.z
    if not 0<=z<ct.shape[0]: raise ValueError(f'slice must be in [0,{ct.shape[0]-1}]')
    raw=np.asarray(ct[z]); sample=np.asarray(ct[::8,::8,::8],dtype=np.float32); lo,hi=np.percentile(sample,[1,99.5]); g=np.clip((raw.astype(np.float32)-lo)*255/(hi-lo),0,255).astype(np.uint8)
    def getmask(name):
        mm=tifffile.memmap(STAGE/name); m=np.asarray(mm[min(z//2,mm.shape[0]-1)],dtype=bool); return np.repeat(np.repeat(m,2,0),2,1)[:raw.shape[0],:raw.shape[1]]
    miss,exc=getmask('missing_material_mask_ds4.tif'),getmask('excess_material_mask_ds4.tif')
    inv={}
    with (STAGE/'all_struts_inventory.csv').open(newline='') as f:
        for r in csv.DictReader(f): inv[int(r['strut_id'])]=(r['classification'],np.array([float(r['centroid_x_um']),float(r['centroid_y_um']),float(r['centroid_z_um'])])/float(r['voxel_size_um']))
    ids=np.array(sorted(inv)); tree=cKDTree(np.array([inv[i][1] for i in ids])*[1,1,1.5])
    cad=json.load(CAD.open()); j={int(x['id']):np.array(x['position'],float) for x in cad['junctions']}; visible=[]
    for s in cad['struts']:
        p0,p1=j[int(s['junction0'])],j[int(s['junction1'])]
        if min(p0[2],p1[2])-4<=z<=max(p0[2],p1[2])+4: visible.append((int(s['id']),p0,p1,inv.get(int(s['id']),('Nominal',))[0]))
    comp,n=ndimage.label(miss); records=[]
    for rid,sl in enumerate(ndimage.find_objects(comp),1):
        if sl is None: continue
        y,x=np.where(comp[sl]==rid); y+=sl[0].start; x+=sl[1].start
        if len(x): records.append((y,x))
    labels=set(); rows=[]
    if records: dist,ix=tree.query(np.array([[x.mean(),y.mean(),z] for y,x in records])*[1,1,1.5],k=min(5,len(ids)),distance_upper_bound=30)
    for ri,(y,x) in enumerate(records):
        cand=[(int(ids[k]),inv[int(ids[k])][0]) for d,k in zip(np.atleast_1d(dist[ri]),np.atleast_1d(ix[ri])) if np.isfinite(d) and k<len(ids)] or [(-1,'Unmatched')]; color='purple' if np.any(exc[y,x]) else 'red'; cx,cy=round(float(x.mean()),3),round(float(y.mean()),3)
        for sid,cls in cand: labels.add(sid); rows.append({'slice_index':z,'strut_id':sid,'classification':cls,'centroid_x':cx,'centroid_y':cy,'centroid_z':z,'region_color':color})
    base=Image.fromarray(np.repeat(g[...,None],3,2),'RGB'); ov=np.array(base); ov[miss]=(0.35*ov[miss]+0.65*np.array([255,40,40])).astype(np.uint8); ov[exc]=(0.35*ov[exc]+0.65*np.array([255,220,30])).astype(np.uint8); ov[miss&exc]=[255,0,255]
    cadim=base.copy(); d=ImageDraw.Draw(cadim)
    for sid,p0,p1,cls in visible:
        col={'Missing_Intentional':(60,150,255),'Missing_Unintentional':(255,100,0)}.get(cls,(80,255,80)); d.line((p0[0],p0[1],p1[0],p1[1]),fill=col,width=1); mx,my=int((p0[0]+p1[0])/2),int((p0[1]+p1[1])/2); d.text((mx,my-7),str(sid),fill=(0,0,0) if sid in labels else col)
    outim=Image.new('RGB',(2511,843),'white')
    for i,p in enumerate([base,Image.fromarray(ov),cadim]): outim.paste(p,(837*i,28))
    d=ImageDraw.Draw(outim); d.text((8,7),'Raw grayscale CT',fill='black'); d.text((845,7),'Existing missing/excess overlay',fill='black'); d.text((1682,7),'Registered CAD / expected struts',fill='black'); d.text((8,824),'red=missing  yellow=excess  purple=overlap',fill='black')
    ip=OUT/f'brian_slice_{z:04d}_validation.png'; outim.save(ip); cp=OUT/f'brian_slice_{z:04d}_defect_regions.csv'
    with cp.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=['slice_index','strut_id','classification','centroid_x','centroid_y','centroid_z','region_color']); w.writeheader(); w.writerows(rows)
    print(f'slice={z} missing_pixels={int(miss.sum())} excess_pixels={int(exc.sum())} visible_cad_struts={len(visible)} matched_rows={len(rows)}'); print(f'image={ip}'); print(f'csv={cp}')
if __name__=='__main__': main()
