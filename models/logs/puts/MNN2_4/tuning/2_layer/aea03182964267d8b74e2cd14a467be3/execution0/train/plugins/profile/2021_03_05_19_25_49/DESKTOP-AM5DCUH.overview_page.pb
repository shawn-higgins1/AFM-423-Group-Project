�	S�
c%7@S�
c%7@!S�
c%7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-S�
c%7@�����4@1��<,��?A.��e�O�?I�{�< @*	     �E@2U
Iterator::Model::ParallelMapV2��0�*�?!�w�q;@)��0�*�?1�w�q;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!NYS֔5;@)��ׁsF�?1B_��7@:Preprocessing2F
Iterator::Model46<��?!w�qGE@)9��v��z?1��#�;.@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��~j�t�?!}A_�6@)�+e�Xw?1�/���*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!kʚ���!@)ŏ1w-!o?1kʚ���!@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�D���J�?!�;⎸L@)��_�Le?1����/@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!0����@)��H�}]?10����@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��_�L�?!����/8@)��H�}M?10���� @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��ӫS�X@Q=k�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����4@�����4@!�����4@      ��!       "	��<,��?��<,��?!��<,��?*      ��!       2	.��e�O�?.��e�O�?!.��e�O�?:	�{�< @�{�< @!�{�< @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��ӫS�X@y=k�?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�f�g�?!�f�g�?"5
sequential/dense/MatMulMatMul*���>�?!#)i�>�?0"C
%gradient_tape/sequential/dense/MatMulMatMul����7�?!|�@�O��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul����7�?!i��w>��?0"7
sequential/dense_1/MatMulMatMul����7�?!V��Y-��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�"\h�E�?!�*���Q�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�"\h�E�?!������?"7
sequential/dense_2/MatMulMatMul�"\h�E�?!��]��1�?0"K
$Adam/Adam/update_4/ResourceApplyAdamResourceApplyAdam����7�?!d��X"5�?"!
Adam/PowPow����7�?!�3�8�?Q      Y@Y>����/@aX�i��U@q���o�X@yv`��O�?"�
both�Your program is POTENTIALLY input-bound because 89.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 