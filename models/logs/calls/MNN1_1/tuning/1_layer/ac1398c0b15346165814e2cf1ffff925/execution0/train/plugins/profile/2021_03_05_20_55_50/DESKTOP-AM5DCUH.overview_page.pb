�	F�~�K5@F�~�K5@!F�~�K5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-F�~�K5@-��2@1�"�J %�?A��ʡE�?Iݳ�Ѳ @*	33333�J@2U
Iterator::Model::ParallelMapV2	�^)ː?!>G�D=m>@)	�^)ː?1>G�D=m>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!����8@)A��ǘ��?1��K3��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�������?!@赓�07@)��y�):�?1~U�R�0@:Preprocessing2F
Iterator::Model$����ۗ?!DPu�5�E@)lxz�,C|?1��$\�)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!KFU�@)��H�}m?1KFU�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipK�=�U�?!���T�bL@)F%u�k?1�Cc}@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!����S@)/n��b?1����S@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy�&1��?!f�'�Y�9@)�~j�t�X?14鏃qC@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�2�ŝ�X@QJ�f?��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	-��2@-��2@!-��2@      ��!       "	�"�J %�?�"�J %�?!�"�J %�?*      ��!       2	��ʡE�?��ʡE�?!��ʡE�?:	ݳ�Ѳ @ݳ�Ѳ @!ݳ�Ѳ @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�2�ŝ�X@yJ�f?��?�"5
sequential/dense/MatMulMatMul����?!����?0"C
%gradient_tape/sequential/dense/MatMulMatMul�F���Ǧ?!��b���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchxr���?!�����[�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulxr���?!4U��(��?"7
sequential/dense_1/MatMulMatMulxr���?!#��L��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam���ʈ�?!��1����?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad���ʈ�?!��g��:�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul���ʈ�?!`؝U���?0"
Abs_1Abs��"0͎?!�X�W���?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam��"0͎?!�l�,�R�?Q      Y@Y��/Ċ�0@a�	�N]�T@q;���rX@y��Qg�
�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 