�	���w��5@���w��5@!���w��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���w��5@}�b�:d3@1�*�����?A��ͪ�զ?IIH�m�I @*	������G@2U
Iterator::Model::ParallelMapV2lxz�,C�?!	N�<=@)lxz�,C�?1	N�<=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatA��ǘ��?!Jݗ�V�7@)HP�sׂ?1[4��}3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateZd;�O��?!�A�I]8@)vq�-�?1	N꾼0@:Preprocessing2F
Iterator::Model��ZӼ�?!`[4�E@)F%u�{?1lE�pR�+@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice��H�}m?!���c+�@)��H�}m?1���c+�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipS�!�uq�?!����cL@)�����g?1�c+���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�/��@)ŏ1w-!_?1�/��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!'u_;@)��_�LU?1��/��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIm�L�[�X@Q�I�Y;��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	}�b�:d3@}�b�:d3@!}�b�:d3@      ��!       "	�*�����?�*�����?!�*�����?*      ��!       2	��ͪ�զ?��ͪ�զ?!��ͪ�զ?:	IH�m�I @IH�m�I @!IH�m�I @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qm�L�[�X@y�I�Y;��?�"5
sequential/dense/MatMulMatMule2�q\�?!e2�q\�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�,���ä?!�/O�1�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�7lw���?!�=*�m��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�7lw���?!̥�T��?"7
sequential/dense_1/MatMulMatMul�7lw���?!�,�����?0"!
Adam/PowPow�,���Ô?!f2�q\�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�,���Ô?! 8lw���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam2C�(�%�?!3��9N��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam2C�(�%�?!f@�����?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam2C�(�%�?!�����?Q      Y@Y��/Ċ�0@a�	�N]�T@q��,��zX@yt��U�?"�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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