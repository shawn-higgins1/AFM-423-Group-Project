�	��v�
h6@��v�
h6@!��v�
h6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��v�
h6@p���3@1�\QJV�?A,e�X�?I�g��@*	gffff�J@2U
Iterator::Model::ParallelMapV2Έ����?!�4ʯ�rA@)Έ����?1�4ʯ�rA@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM�St$�?!o2�ad35@)Έ����?1�4ʯ�r1@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�(��0�?!;�;�7@)�� �rh�?1�k�j�/@:Preprocessing2F
Iterator::Model�(��0�?!;�;�G@)�~j�t�x?1��%-��&@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!����@)ŏ1w-!o?1����@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO��e�c�?!�N��N�J@)y�&1�l?11�V�3D@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�쇑�@)����Mb`?1�쇑�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�Pk�w�?!P�0,:@)-C��6Z?1���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�58*�X@Q��1`u�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	p���3@p���3@!p���3@      ��!       "	�\QJV�?�\QJV�?!�\QJV�?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	�g��@�g��@!�g��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�58*�X@y��1`u�?�"5
sequential/dense/MatMulMatMul���1֦?!���1֦?0"C
%gradient_tape/sequential/dense/MatMulMatMulx"�HL�?!��7=G��?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradk��_�?!f�/�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulk��_�?!Re� �?"7
sequential/dense_1/MatMulMatMulk��_�?!y"�HL�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamx"�HL�?!���1��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchx"�HL�?!k��_�?"$
truedivRealDiv�EQ�X�?!�E���?"E
'gradient_tape/sequential/dense_1/MatMulMatMulX��|Hu�?!H�����?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam����r�?!���<��?Q      Y@Y��/Ċ�0@a�	�N]�T@qh�93e�X@y܅�ᆷ�?"�
both�Your program is POTENTIALLY input-bound because 88.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.1% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 