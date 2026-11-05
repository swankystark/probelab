const demo$=(id)=>document.getElementById(id);
const stageSpecs=[
  {start:0,end:25,title:'Problem introduction',time:'00:00–00:25',focus:null},
  {start:25,end:45,title:'Introducing ProbeLab',time:'00:25–00:45',focus:null},
  {start:45,end:75,title:'Observation',time:'00:45–01:15',focus:'demo-observation'},
  {start:75,end:100,title:'Bayesian inference',time:'01:15–01:40',focus:'demo-model'},
  {start:100,end:130,title:'Active experiment selection',time:'01:40–02:10',focus:'demo-action'},
  {start:130,end:155,title:'Prediction and validation',time:'02:10–02:35',focus:'demo-prediction'},
  {start:155,end:180,title:'Closing',time:'02:35–03:00',focus:'demo-closing'},
];
let replay=null, startedAt=0, elapsedBeforePause=0, ticker=null, paused=false, liveStarted=false, liveId=null, activeStage=-1;
const fmtClock=(seconds)=>`${Math.floor(seconds/60)}:${String(Math.floor(seconds%60)).padStart(2,'0')}`;
const demoApi=async(path)=>{const response=await fetch(path);const body=await response.json();if(!response.ok)throw new Error(body.detail||`Request failed (${response.status})`);return body};
const drawDemoTrajectory=(target,records)=>{
  if(!records?.length){target.textContent='No saved angular measurements.';return}
  const values=records.map((record)=>record.angle_deg), min=Math.min(...values), max=Math.max(...values), span=Math.max(max-min,1);
  const points=records.map((record,index)=>`${(index/Math.max(records.length-1,1))*100},${92-((record.angle_deg-min)/span)*80}`).join(' ');
  target.innerHTML=`<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true"><line x1="0" y1="92" x2="100" y2="92" stroke="currentColor" opacity=".25"/><polyline points="${points}" fill="none" stroke="var(--lime)" stroke-width="1.8" vector-effect="non-scaling-stroke"/></svg><small>${min.toFixed(1)}° to ${max.toFixed(1)}° · ${records.length} measured samples</small>`;
};
const drawPosterior=(posterior)=>{
  const canvas=demo$('demo-posterior'),ctx=canvas.getContext('2d');
  if(!posterior?.weights?.length){canvas.insertAdjacentHTML('afterend','<p class="demo-caption">Posterior grid unavailable in this saved run.</p>');return}
  const rows=posterior.weights.length,cols=posterior.weights[0].length,values=posterior.weights.flat(),peak=Math.max(...values);
  canvas.width=cols;canvas.height=rows;const image=ctx.createImageData(cols,rows);
  for(let y=0;y<rows;y++)for(let x=0;x<cols;x++){const ratio=peak>0?posterior.weights[y][x]/peak:0,intensity=Math.max(0,(Math.log10(Math.max(ratio,1e-10))+10)/10),p=(y*cols+x)*4;image.data[p]=Math.round(24+intensity*176);image.data[p+1]=Math.round(34+intensity*214);image.data[p+2]=Math.round(29+intensity*52);image.data[p+3]=255}
  ctx.putImageData(image,0,0);canvas.setAttribute('aria-label',`Log-scaled saved posterior weights: length ${posterior.length_grid_m[0].toFixed(2)} to ${posterior.length_grid_m.at(-1).toFixed(2)} metres; damping ${posterior.damping_grid_s_minus_1[0].toFixed(3)} to ${posterior.damping_grid_s_minus_1.at(-1).toFixed(3)} per second`);
};
const drawTrackOverlay=()=>{
  const canvas=demo$('demo-track-overlay'),video=demo$('demo-run-video');if(!replay||!video.videoWidth||!video.videoHeight)return;
  const rect=video.getBoundingClientRect(),dpr=window.devicePixelRatio||1;canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
  const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  const scale=Math.min(rect.width/video.videoWidth,rect.height/video.videoHeight),w=video.videoWidth*scale,h=video.videoHeight*scale,ox=(rect.width-w)/2,oy=(rect.height-h)/2,project=(x,y)=>[ox+x*scale,oy+y*scale];
  const rawPivot=replay.annotation?.pivot,pivot=Array.isArray(rawPivot)?{x:rawPivot[0],y:rawPivot[1]}:rawPivot;if(pivot){const [px,py]=project(pivot.x,pivot.y);ctx.fillStyle='#8fb6ff';ctx.beginPath();ctx.arc(px,py,4,0,Math.PI*2);ctx.fill()}
  const tracks=replay.tracking.tracks||[];if(!tracks.length)return;
  ctx.strokeStyle='#c8ff5a';ctx.lineWidth=2;ctx.beginPath();tracks.forEach((track,index)=>{const [x,y]=project(track.centroid.x,track.centroid.y);index?ctx.lineTo(x,y):ctx.moveTo(x,y)});ctx.stroke();
  const targetTime=Math.max(replay.sampling.start_s,Math.min(replay.sampling.end_s,video.currentTime));let nearest=tracks[0];for(const track of tracks)if(Math.abs(track.timestamp_s-targetTime)<Math.abs(nearest.timestamp_s-targetTime))nearest=track;
  const [x,y]=project(nearest.centroid.x,nearest.centroid.y);ctx.fillStyle='#c8ff5a';ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fill();ctx.strokeStyle='#101310';ctx.lineWidth=2;ctx.stroke();
};
const renderEvidence=(item)=>{
  replay=item;demo$('demo-run-video').src=item.video_url;demo$('demo-source-video').src=item.video_url;
  const tracking=item.tracking,geometry=item.geometry_quality,comparison=item.comparison,fit=item.fit;
  demo$('demo-observation-meta').textContent=`object ${tracking.object_id} · ${tracking.frames_tracked} sampled frames · source ${item.sampling.start_s.toFixed(2)}–${item.sampling.end_s.toFixed(2)} s`;
  demo$('demo-tracking').textContent=`SAM 3.1 · ${tracking.frames_with_masks}/${tracking.frames_tracked} masks`;
  demo$('demo-confidence').textContent=`${(tracking.confidence*100).toFixed(1)}% mean`;
  demo$('demo-radius').textContent=`${geometry.radius.median_px.toFixed(1)} px · CV ${(geometry.radius.cv*100).toFixed(2)}%`;
  demo$('demo-coverage').textContent=`${(geometry.coverage_fraction*100).toFixed(0)}% · pixel geometry PASS`;
  demo$('demo-comparison').textContent=comparison?`Independent CSV comparison · RMSE ${comparison.angular_rmse_deg.toFixed(2)}° · max ${comparison.angular_max_error_deg.toFixed(2)}° · offset +${comparison.estimated_time_offset_s.toFixed(3)} s · overlap ${comparison.overlap_frame_count} frames / ${comparison.overlap_duration_s.toFixed(3)} s`:'No independent CSV comparison is saved.';
  drawDemoTrajectory(demo$('demo-observed-chart'),geometry.pixel_geometry);drawPosterior(item.posterior);
  demo$('demo-adequacy').textContent=fit.adequacy.toUpperCase();demo$('demo-length').textContent=`${fit.length_m.toFixed(2)} m`;
  demo$('demo-length-interval').textContent=`credible interval ${fit.length_interval_m.map((v)=>v.toFixed(2)).join('–')} m`;
  demo$('demo-damping').textContent=`${fit.damping_s_minus_1.toFixed(4)} s⁻¹`;demo$('demo-damping-interval').textContent=`credible interval ${fit.damping_interval_s_minus_1.map((v)=>v.toFixed(4)).join('–')} s⁻¹`;
  demo$('demo-residual').textContent=`${fit.residual_rms_deg.toFixed(2)}°`;
  const velocityInterval=fit.initial_velocity_interval_rad_s;
  demo$('demo-initial-velocity').textContent=Number.isFinite(fit.initial_velocity_rad_s)?`${fit.initial_velocity_rad_s.toFixed(3)} rad/s${velocityInterval?` · 68% ${velocityInterval.map((v)=>v.toFixed(3)).join('–')}`:''}`:'Not estimated';
  const velocityBounds=fit.initial_velocity_prior_bounds_rad_s;
  demo$('demo-inference-status').textContent=velocityBounds?`Production Bayesian posterior from ${tracking.frames_tracked} saved SAM angles; start-rate prior ${velocityBounds[0].toFixed(2)} to ${velocityBounds[1].toFixed(2)} rad/s. Independent CSV excluded.`:'Production Bayesian posterior from saved SAM angles only. Independent CSV excluded.';

  const selector=item.selector?.angle_deg>0&&fit.adequacy==='good'?item.selector:null;
  if(selector){
    demo$('demo-eig-state').textContent=`Recommended · ${selector.angle_deg.toFixed(0)}°`;
    demo$('demo-eig-rationale').textContent=selector.rationale;
    demo$('demo-eig-value').textContent=`Expected information gain ${selector.eig_nats.toFixed(3)} nats · ${selector.duration_s.toFixed(1)} s`;
    const scores=selector.scores||[];
    demo$('demo-eig-scores').textContent=scores.length?`Candidate scores: ${scores.map((score)=>`${score.angle_deg.toFixed(0)}° ${score.eig_nats.toFixed(2)}`).join(' · ')}`:'Candidate score breakdown unavailable.';
  }else{
    demo$('demo-eig-state').textContent=`Action withheld · ${fit.adequacy==='good'?'no saved selector result':'fit inadequate'}`;
    demo$('demo-eig-rationale').textContent=`The production fit residual is ${fit.residual_rms_deg.toFixed(2)}°; the existing adequacy threshold is 3.00°. No unsupported action is shown.`;
    demo$('demo-eig-value').textContent='EIG recommendation unavailable for this saved run.';
    demo$('demo-eig-scores').textContent='';
  }

  const museRecord=item.muse||{},museText=museRecord.text?.trim()||'';
  const museMatchesRun=museRecord.session_id===item.session_id&&/^[a-f0-9]{64}$/.test(museRecord.evidence_fingerprint||'');
  const museCompleted=String(museRecord.status||'').toLowerCase()==='completed'&&museText.length>0&&museMatchesRun;
  if(museCompleted){
    demo$('demo-muse-status').textContent=`Completed · ${museRecord.model} · ${museRecord.response_text_length} chars`;
    demo$('demo-muse-text').textContent=museText;
  }else if(museRecord.status==='Failed'||museRecord.status==='Empty response'){
    demo$('demo-muse-status').textContent=`${museRecord.status} · ${museRecord.model||'Muse Spark'}`;
    demo$('demo-muse-text').textContent=museRecord.error_detail||'Muse returned an empty response; no interpretation is saved.';
  }else if(museRecord.status==='Incomplete response'){
    demo$('demo-muse-status').textContent=`Incomplete · ${museRecord.model||'Muse Spark'} · finish=${museRecord.finish_reason||'unknown'}`;
    demo$('demo-muse-text').textContent=`Partial API text: “${museText||'no visible text'}”. ${museRecord.error_detail||'The response was truncated and is not a complete interpretation.'}`;
  }else if(museRecord.status==='Completed'&&!museMatchesRun){
    demo$('demo-muse-status').textContent='Response provenance mismatch';
    demo$('demo-muse-text').textContent='The saved response could not be verified against this session and is not displayed as its interpretation.';
  }else{
    demo$('demo-muse-status').textContent='Not invoked for this run';
    demo$('demo-muse-text').textContent=`Numerical interpretation (not a Muse response): ${item.numerical_interpretation||'Muse Spark was not called for this cached run.'}`;
  }

  const prediction=item.frozen_prediction,validation=item.validation||{};
  const hasPrediction=Array.isArray(prediction?.angles_deg)&&prediction.angles_deg.length>1;
  const validationAngles=validation.angles_deg||validation.actual_angles_deg||[];
  if(hasPrediction){
    const series=[{values:prediction.angles_deg,color:'var(--blue)'}];
    if(validationAngles.length>1)series.push({values:validationAngles,color:'var(--lime)'});
    demoChart(demo$('demo-prediction-chart'),series);
    demo$('demo-prediction-summary').textContent=`Frozen forecast · ${prediction.angle_deg.toFixed(0)}° release · ${prediction.duration_s.toFixed(1)} s · model-generated, not observed.`;
    demo$('demo-prediction-status').textContent=validationAngles.length>1?'PREDICTION VALIDATED':'FROZEN · AWAITING VALIDATION';
  }else{
    demo$('demo-prediction-chart').textContent='No frozen prediction is available.';
    demo$('demo-prediction-summary').textContent='No frozen prediction is available.';
    demo$('demo-prediction-status').textContent='AWAITING VALIDATION';
  }
  if(validationAngles.length>1){
    demo$('demo-validation-metrics').textContent=`Observed validation is available · RMSE ${Number(validation.rmse_deg).toFixed(2)}° · max error ${Number(validation.max_error_deg).toFixed(2)}°.`;
  }else{
    demo$('demo-validation-metrics').textContent=validation.message||'No second independent observation is available; the frozen curve remains an unvalidated prediction.';
  }
  demo$('demo-report-note').textContent='Pixel tracking and geometry passed. Metric calibration is unavailable, so production handoff remains blocked. Synthetic simulator validation only; no physical validation claim.';
};
const stopPlayback=()=>{if(ticker){clearInterval(ticker);ticker=null}demo$('demo-run-video').pause()};
const showChooser=()=>{stopPlayback();document.body.classList.add('demo-mode');demo$('launch-home').classList.remove('hidden');demo$('demo-home').classList.add('hidden');demo$('demo-dashboard').classList.add('hidden');demo$('setup-view').classList.add('hidden');demo$('workspace-view').classList.add('hidden');demo$('mode-toggle').textContent='Choose workflow';demo$('workflow-context').textContent='CHOOSE A WORKFLOW'};
const updateStage=(elapsed)=>{
  const stageIndex=stageSpecs.findIndex((item)=>elapsed>=item.start&&elapsed<item.end),stage=stageSpecs[stageIndex<0?stageSpecs.length-1:stageIndex];
  if(stageIndex!==activeStage){activeStage=stageIndex;const video=demo$('demo-run-video');if(stageIndex===2&&replay){if(video.readyState>=1){video.currentTime=replay.sampling.start_s;video.playbackRate=(replay.sampling.end_s-replay.sampling.start_s)/(stage.end-stage.start);video.play().catch(()=>{})}else pendingStageSeek=true}else if(!video.paused)video.pause()}
  demo$('demo-stage-time').textContent=stage.time;demo$('demo-stage-title').textContent=stage.title;demo$('demo-intro-overlay').classList.toggle('visible',elapsed<25);demo$('demo-closing').classList.toggle('visible',elapsed>=160);
  document.querySelectorAll('.demo-card').forEach((card)=>{const active=card.id===stage.focus;card.classList.toggle('is-active',active);card.classList.toggle('is-dimmed',Boolean(stage.focus)&&stage.focus!=='demo-closing'&&!active)});
  demo$('demo-progress-fill').style.width=`${Math.min(elapsed/180*100,100)}%`;
  if(elapsed>=45&&elapsed<75)drawTrackOverlay();
  if(elapsed>=180){stopPlayback();elapsedBeforePause=180;paused=true;demo$('demo-pause').textContent='Play again';demo$('demo-clock').textContent='3:00 / 3:00';demo$('demo-progress-fill').style.width='100%';return}
  demo$('demo-clock').textContent=`${fmtClock(elapsed)} / 3:00`;
};
const playbackTick=()=>updateStage(Math.min(elapsedBeforePause+(performance.now()-startedAt)/1000,180));
const startPlayback=(reset=true)=>{stopPlayback();if(reset){elapsedBeforePause=0;activeStage=-1}paused=false;startedAt=performance.now();demo$('demo-pause').textContent='Pause';ticker=setInterval(playbackTick,100);playbackTick()};
const togglePause=()=>{if(paused){if(elapsedBeforePause>=180){startPlayback(true);return}startedAt=performance.now();paused=false;ticker=setInterval(playbackTick,100);if(activeStage===2)demo$('demo-run-video').play().catch(()=>{});demo$('demo-pause').textContent='Pause';return}if(!ticker)return;elapsedBeforePause=Math.min(elapsedBeforePause+(performance.now()-startedAt)/1000,180);paused=true;clearInterval(ticker);ticker=null;demo$('demo-run-video').pause();demo$('demo-pause').textContent='Resume'};
demo$('launch-live').addEventListener('click',()=>{stopPlayback();document.body.classList.remove('demo-mode');demo$('launch-home').classList.add('hidden');demo$('demo-home').classList.add('hidden');demo$('demo-dashboard').classList.add('hidden');demo$('setup-view').classList.toggle('hidden',liveStarted);demo$('workspace-view').classList.toggle('hidden',!liveStarted);demo$('mode-toggle').textContent='Choose workflow';demo$('workflow-context').textContent='LIVE EXPERIMENT · NEW INVESTIGATION'});
demo$('launch-cached').addEventListener('click',()=>{document.body.classList.add('demo-mode');demo$('launch-home').classList.add('hidden');demo$('demo-home').classList.remove('hidden');demo$('demo-dashboard').classList.add('hidden');demo$('setup-view').classList.add('hidden');demo$('workspace-view').classList.add('hidden');demo$('mode-toggle').textContent='Choose workflow';demo$('workflow-context').textContent='CACHED DEMO REPLAY'});
demo$('demo-back-chooser').addEventListener('click',showChooser);
demo$('demo-play-start').addEventListener('click',()=>{if(!replay)return;document.body.classList.add('demo-mode');demo$('demo-home').classList.add('hidden');demo$('demo-dashboard').classList.remove('hidden');demo$('workflow-context').textContent='CACHED DEMO REPLAY · SAVED EVIDENCE';startPlayback(true)});
demo$('demo-pause').addEventListener('click',togglePause);demo$('demo-restart').addEventListener('click',()=>startPlayback(true));
demo$('demo-muse-once').addEventListener('click',async()=>{
  const button=demo$('demo-muse-once');
  if(!replay?.session_id||button.disabled)return;
  button.disabled=true;button.textContent='Requesting Muse once…';
  demo$('demo-muse-call-status').textContent='Sending only this run’s SAM-derived measurements and numerical outputs to Muse Spark.';
  try{
    const result=await demoApiPost(`/api/demo/synthetic-replay/${replay.session_id}/muse`,{});
    replay.muse=result;renderEvidence(replay);
    if(result.status==='Completed'&&result.text){button.textContent='Muse response saved';demo$('demo-muse-call-status').textContent=`Actual ${result.model} response saved for this session (${result.response_text_length} characters). Playback will use this saved response without another model call.`}
    else{button.textContent='Muse attempt recorded';demo$('demo-muse-call-status').textContent=`${result.status}: ${result.error_detail||result.text||'No response text was returned.'} No second request will be made.`}
  }catch(error){button.textContent='Muse attempt may be recorded';demo$('demo-muse-call-status').textContent=`Request outcome unavailable · ${error.message}. Check the saved session diagnostics before taking further action; playback will not retry.`}
});
demo$('mode-toggle').addEventListener('click',showChooser);
let pendingStageSeek=false;
demo$('demo-run-video').addEventListener('loadedmetadata',()=>{drawTrackOverlay();if(pendingStageSeek&&replay){const video=demo$('demo-run-video');video.currentTime=replay.sampling.start_s;video.playbackRate=(replay.sampling.end_s-replay.sampling.start_s)/30;pendingStageSeek=false;if(!paused)video.play().catch(()=>{})}});
demo$('demo-run-video').addEventListener('timeupdate',drawTrackOverlay);window.addEventListener('resize',drawTrackOverlay);
demoApi('/api/demo/synthetic-replay').then((item)=>{renderEvidence(item);demo$('demo-load-status').textContent='Cached real run ready · playback makes no SAM or Muse requests.';demo$('demo-play-start').disabled=false;const attempted=Boolean(item.muse?.invocation_metadata?.attempted);demo$('demo-muse-once').disabled=attempted;if(attempted){demo$('demo-muse-once').textContent=item.muse.status==='Completed'?'Muse response saved':'Muse attempt recorded';demo$('demo-muse-call-status').textContent=item.muse.status==='Completed'?`Saved ${item.muse.model} response is linked to this session; playback will not call Muse.`:`${item.muse.status}: ${item.muse.error_detail||item.muse.text||'No response text was returned.'} The one-time attempt is already recorded.`}}).catch((error)=>{demo$('demo-load-status').textContent=`Cached run unavailable · ${error.message}`;demo$('demo-play-start').disabled=true});

const demoStage=(name)=>{const rail=demo$('workflow-rail');if(!rail)return;const order=['observation','inference','muse','prediction','validation'],current=order.indexOf(name);rail.querySelectorAll('[data-stage]').forEach((item,index)=>{item.classList.toggle('complete',index<current);item.classList.toggle('active',index===current)})};
const demoChart=(target,series)=>{const valid=series.filter((item)=>item.values?.length>1);if(!valid.length){target.textContent='Trajectory unavailable';return}const all=valid.flatMap((item)=>item.values),lo=Math.min(...all),hi=Math.max(...all),span=Math.max(hi-lo,1),paths=valid.map((item)=>`<polyline points="${item.values.map((value,index)=>`${index/(item.values.length-1)*100},${90-(value-lo)/span*78}`).join(' ')}" fill="none" stroke="${item.color}" stroke-width="1.5" vector-effect="non-scaling-stroke"/>`).join('');target.innerHTML=`<svg viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">${paths}</svg>`};
const originalSetWorkspace=window.setWorkspace;
window.setWorkspace=(item)=>{liveId=item.id;liveStarted=true;stopPlayback();document.body.classList.remove('demo-mode');demo$('launch-home').classList.add('hidden');demo$('demo-home').classList.add('hidden');demo$('demo-dashboard').classList.add('hidden');demo$('workflow-context').textContent='LIVE EXPERIMENT · ACTIVE INVESTIGATION';originalSetWorkspace(item);demo$('setup-view').classList.add('hidden');demoStage(item.observations?.length?'inference':'observation');demo$('retry-muse').classList.toggle('hidden',item.agent_status==='Completed'||item.agent_status==='Pending');const prediction=item.prediction;if(prediction){demoChart(demo$('prediction-chart'),[{values:prediction.angles_deg,color:'var(--blue)'}]);demo$('prediction-state').textContent='Frozen · awaiting validation';demo$('validation-angle').value=prediction.angle_deg;demo$('validation-duration').value=prediction.duration_s}else{demo$('prediction-state').textContent='Awaiting a valid fit';demo$('prediction-chart').textContent='No supported prediction yet.'}};
const demoApiPost=async(path,payload)=>{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}),body=await response.json();if(!response.ok)throw new Error(body.detail||`Request failed (${response.status})`);return body};
window.finishDiscovery=async(item)=>{demoStage('inference');if(item.status!=='experiment recommended'||!(item.recommendation.angle_deg>0)){demo$('prediction-state').textContent='No prediction: fit inadequate';demoStage('validation');return}demoStage('muse');if(item.agent_status!=='Completed')demo$('reasoning').textContent=`Agent reasoning unavailable (${item.agent_status}). Numerical results are preserved.`;demoStage('prediction');try{const response=await demoApiPost(`/api/investigations/${item.id}/predict`,{angle_deg:item.recommendation.angle_deg,duration_s:item.recommendation.duration_s});window.setWorkspace(response);demo$('prediction-state').textContent='Frozen · awaiting validation';demo$('prediction-validation-state').textContent='Prediction frozen before the next observation. Upload the resulting physical experiment in Prediction and validation.';demoStage('validation')}catch(error){demo$('prediction-state').textContent=`Prediction failed · ${error.message}`}};
const originalSubmitObservation=window.submitObservation;
window.submitObservation=async(...args)=>{demoStage('observation');const item=await originalSubmitObservation(...args);if(item&&args[0].includes('/observe-guided'))await window.finishDiscovery(item);return item};
demo$('retry-muse').addEventListener('click',async()=>{demo$('retry-muse').disabled=true;demo$('reasoning').textContent='Retrying Muse from saved numerical evidence…';try{const response=await fetch(`/api/investigations/${liveId}/reason`,{method:'POST'}),item=await response.json();if(!response.ok)throw new Error(item.detail||`Request failed (${response.status})`);window.setWorkspace(item)}catch(error){demo$('reasoning').textContent=`Agent reasoning unavailable · ${error.message}`}finally{demo$('retry-muse').disabled=false}});
demo$('validation-input').addEventListener('change',()=>demoStage('validation'));
demo$('start-btn').addEventListener('click',()=>{liveStarted=true;document.body.classList.remove('demo-mode')});
demo$('video-input').addEventListener('change',(event)=>{const file=event.target.files[0];if(!file)return;const video=demo$('source-preview');video.src=URL.createObjectURL(file);video.hidden=false});
