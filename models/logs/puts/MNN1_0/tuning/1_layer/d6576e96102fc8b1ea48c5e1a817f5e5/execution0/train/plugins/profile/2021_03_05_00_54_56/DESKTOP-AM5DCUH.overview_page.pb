�	Zd;ߏ6@Zd;ߏ6@!Zd;ߏ6@	���i0�?���i0�?!���i0�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Zd;ߏ6@��s��4@1P��5&�?A,e�X�?IBZc�	a@Y�26t�?p?*	������M@2U
Iterator::Model::ParallelMapV2����Mb�?!��е��:@)����Mb�?1��е��:@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenatejM�?!���H�?@)�ZӼ��?1*#���7@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatS�!�uq�?!�~xKr6@)A��ǘ��?1�F�̗2@:Preprocessing2F
Iterator::Model
ףp=
�?!�����B@)9��v��z?1Ь���%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicen��t?!��UXj @)n��t?1��UXj @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipΈ����?!t*) �'O@)-C��6j?1�C���p@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!d�I�@)HP�s�b?1d�I�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�0�*�?!��q�@A@)�~j�t�X?1��\��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���i0�?I�����X@Q(������?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��s��4@��s��4@!��s��4@      ��!       "	P��5&�?P��5&�?!P��5&�?*      ��!       2	,e�X�?,e�X�?!,e�X�?:	BZc�	a@BZc�	a@!BZc�	a@B      ��!       J	�26t�?p?�26t�?p?!�26t�?p?R      ��!       Z	�26t�?p?�26t�?p?!�26t�?p?b      ��!       JGPUY���i0�?b q�����X@y(������?�"5
sequential/dense/MatMulMatMul�X����?!�X����?0"C
%gradient_tape/sequential/dense/MatMulMatMul���G��?!^6-���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchZF�����?!�P�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulZF�����?!���EI��?"7
sequential/dense_1/MatMulMatMulZF�����?!�����?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad{k�_�?!��(���?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam9!{�ˎ?!���C��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam9!{�ˎ?!'g�j}�?"!
Adam/PowPow9!{�ˎ?!;��i�?">
AssignAddVariableOp_9AssignAddVariableOp9!{�ˎ?!O���V�?Q      Y@Y��/Ċ�0@a�	�N]�T@qU���\X@yR#M�5�?"�
both�Your program is POTENTIALLY input-bound because 88.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.2 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 