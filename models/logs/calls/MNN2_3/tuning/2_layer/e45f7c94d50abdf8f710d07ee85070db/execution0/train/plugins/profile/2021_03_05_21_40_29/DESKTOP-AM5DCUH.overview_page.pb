�	O�j�27@O�j�27@!O�j�27@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-O�j�27@�=$|�4@1��H�H�?A�,C��?I�ɧg@*	fffff�G@2U
Iterator::Model::ParallelMapV2y�&1��?!��eo�I=@)y�&1��?1��eo�I=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatA��ǘ��?!�d~�87@)Έ����?1�*���t3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!�q���8@)��ǘ���?1I�G�0@:Preprocessing2F
Iterator::ModelˡE����?!_��,�qE@)9��v��z?1��'�W2+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice_�Q�k?!���s@)_�Q�k?1���s@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�#�����?!�6�u�L@)-C��6j?1P���:�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!�αF" @)��H�}]?1�αF" @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9��v���?!��'�W2;@)�~j�t�X?1���:�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI5�X��X@Q�2�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�=$|�4@�=$|�4@!�=$|�4@      ��!       "	��H�H�?��H�H�?!��H�H�?*      ��!       2	�,C��?�,C��?!�,C��?:	�ɧg@�ɧg@!�ɧg@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q5�X��X@y�2�����?�"5
sequential/dense/MatMulMatMul+ht�?!+ht�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�&&*�?!N��O�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul~" ٜ)�?!��G���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul~" ٜ)�?!����X<�?"7
sequential/dense_1/MatMulMatMul~" ٜ)�?!��30���?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�3�Ek��?!��阍��?"7
sequential/dense_2/MatMulMatMul+ht�?!b��m�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��:���?!�@~1��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���r�*�?!������?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam~" ٜ)�?!���,��?Q      Y@Y>����/@aX�i��U@qK��;zX@y�5B��?"�
both�Your program is POTENTIALLY input-bound because 89.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 