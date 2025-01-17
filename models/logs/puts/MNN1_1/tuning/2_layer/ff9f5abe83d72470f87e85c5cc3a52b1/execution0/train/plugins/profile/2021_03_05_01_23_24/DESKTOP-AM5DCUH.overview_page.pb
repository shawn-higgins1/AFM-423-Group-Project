�	`�;�8@`�;�8@!`�;�8@	ٌz[���?ٌz[���?!ٌz[���?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6`�;�8@�K⬈�5@1�$"����?A+��Χ?I�^ @Y��y�):r?*�����YH@)       =2U
Iterator::Model::ParallelMapV2�?�߾�?!=k���!<@)�?�߾�?1=k���!<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�(��0�?!s��Z�A9@)��ZӼ�?1�BW���4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǺ����?!SN�my�6@)ŏ1w-!?1�|��$6/@:Preprocessing2F
Iterator::Model�0�*�?!Q��_&E@)lxz�,C|?1��9�5V,@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!"@���@)��H�}m?1"@���@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�?Ɯ?!�UT��L@)�~j�t�h?1ƊH�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!��e�?@)�J�4a?1��e�?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!;ǳƊH:@)-C��6Z?1;ǳƊH
@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9ٌz[���?I]A��ɤX@Qt��[���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�K⬈�5@�K⬈�5@!�K⬈�5@      ��!       "	�$"����?�$"����?!�$"����?*      ��!       2	+��Χ?+��Χ?!+��Χ?:	�^ @�^ @!�^ @B      ��!       J	��y�):r?��y�):r?!��y�):r?R      ��!       Z	��y�):r?��y�):r?!��y�):r?b      ��!       JGPUYٌz[���?b q]A��ɤX@yt��[���?�"5
sequential/dense/MatMulMatMulJE ��
�?!JE ��
�?0"C
%gradient_tape/sequential/dense/MatMulMatMul%!9���?!8�~���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul%!9���?!�C9}*r�?"7
sequential/dense_1/MatMulMatMul%!9���?!.�*>Ot�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�<.ˠ?!�����?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchni��!(�?!s	L�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulni��!(�?!C�+I��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad,��-H�?!(O�N��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam%!9���?!Ms�N���?"!
Adam/PowPow%!9���?!��9'���?Q      Y@Y>����/@aX�i��U@qD<��v�W@y���ݟ��?"�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 